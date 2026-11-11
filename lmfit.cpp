#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <torch/torch.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

using namespace torch::indexing;


class fit {
  public:
  fit(const torch::Tensor& _irGPU, const torch::Tensor& _angFreqGPU,
        float _lambda0, float _decreaseFactor, float _increaseFactor,
        float _clampLimit, float _clampValue, uint _baselineSubtractInd,
        uint _startFreqInd) {

    lambda0 = _lambda0;
    decreaseFactor = _decreaseFactor;
    increaseFactor = _increaseFactor;
    clampLimit = _clampLimit;
    clampValue = _clampValue;
    baselineSubtractInd = _baselineSubtractInd;
    startFreqInd = _startFreqInd;

    nFreqs = _irGPU.size(-1);
    if (_angFreqGPU.size(-1) != nFreqs) throw std::invalid_argument("Impulse response and frequency axis have different lengths.");

    irGPU = _irGPU;
    fGPU = _angFreqGPU;

    options = torch::TensorOptions()
              .dtype(_irGPU.dtype())
              .device(_irGPU.device())
              .layout(torch::kStrided)
              .requires_grad(false);

    torch::Tensor sampleLifeTimes = torch::linspace(0.5, 5, 10, options).unsqueeze(-1);
    torch::Tensor selfCalibrations = responseGPU(torch::tensor(1, options), sampleLifeTimes);
    selfCalibrations /= selfCalibrations.index({Ellipsis, Slice(0, 1)});
    selfCalibrations.square_();
    torch::Tensor integrals = torch::sum(selfCalibrations, -1, false).squeeze();
    torch::Tensor moments = torch::sum(selfCalibrations * fGPU, -1, false).squeeze();
    moments /= integrals;

    // fit to the profile TODO reimplement for multi channel IRF
    torch::Tensor independent = torch::stack({torch::ones_like(moments, options),
                                              moments,
                                              moments.square(),
                                              moments.pow(3),
                                              }, -1);
    regressor = torch::matmul(torch::matmul(torch::linalg_inv(torch::matmul(independent.t(), independent)), independent.t()), sampleLifeTimes);

  }

  torch::Tensor run(torch::Tensor& traces, uint nOptSteps, bool returnAmplitude, bool inFreqDomain=false) {
    torch::Tensor objective;
    if (inFreqDomain) {
      if (traces.dtype() != irGPU.dtype() || traces.size(-1) != nFreqs) throw  std::invalid_argument("Frequency Domain traces not of right type and/or size.");
      objective = traces;
    }
    else {
      if (baselineSubtractInd)
        traces -= torch::mean(traces.index({Ellipsis, Slice(0, baselineSubtractInd)}), -1, true);
      
      torch::Tensor transform = torch::fft_rfft(traces, std::nullopt, -1, "backward").index({Ellipsis, Slice(startFreqInd, startFreqInd+nFreqs)});
      objective = torch::abs(transform);
    }

    torch::Tensor objectiveInit = objective.square();
    torch::Tensor peak = objective.index({Ellipsis, 0}).unsqueeze(-1).clone();
    torch::Tensor integrals = torch::sum(objectiveInit, -1, true);
    torch::Tensor moments = torch::sum(objectiveInit * fGPU, -1, true);
    moments /= integrals;
    torch::Tensor tau = regressor.index({0}) + moments * (regressor.index({1}) + moments * (regressor.index({2}) + moments * regressor.index({3})));
    torch::clip_out(tau, tau, 0.1f, 5.5f);

    torch::Tensor lambdai = torch::full_like(peak, lambda0, options);

    for (uint its = 0; its < nOptSteps; ++its) {
      auto response = responseGPUHess(peak, tau, objective);
      auto& [errorVec, gradientPK, gradientLF, hessianPK, hessianLF, hessianPKLF] = response;

      torch::Tensor lambdaDampening = (1 + lambdai);
      hessianPK += (-1.1f * hessianPK) * (hessianPK < 0) + 1e-7f * (hessianPK == 0);
      hessianPK *= lambdaDampening;
      hessianLF += (-1.1f * hessianLF) * (hessianLF < 0) + 1e-7f * (hessianLF == 0);
      hessianLF *= lambdaDampening;

      torch::Tensor hessianNorm = hessianPK * hessianLF;
      torch::Tensor hessianPKLFSq = hessianPKLF.square();
      hessianNorm -= hessianPKLFSq * (hessianNorm != hessianPKLFSq);
      hessianNorm = -1 / hessianNorm;

      torch::Tensor stepDeltaPK = (gradientPK * hessianLF - gradientLF * hessianPKLF) * hessianNorm;
      torch::Tensor stepDeltaLF = (gradientLF * hessianPK - gradientPK * hessianPKLF) * hessianNorm;

      torch::Tensor _peak = peak + stepDeltaPK;
      torch::Tensor _tau = tau + stepDeltaLF;
      torch::Tensor fitVec = responseGPU(_peak, _tau);
      fitVec -= objective;
      fitVec.square_();
      torch::Tensor errorVec2 = torch::mean(fitVec, -1, true);

      torch::Tensor errorImprovement = errorVec - errorVec2;

      torch::Tensor improvedMask = (errorImprovement > 0);
      peak += improvedMask * stepDeltaPK;
      tau += improvedMask * stepDeltaLF;

      lambdai *= decreaseFactor * improvedMask + increaseFactor * ~improvedMask;
      torch::clamp_out(lambdai, lambdai, 1e-7f, 1e7f);
    }

    tau.abs_();
    peak = torch::where(tau < clampLimit, peak, 0); // assume that these are misinterpreted DC traces
    tau  = torch::where(tau < clampLimit, tau, clampValue); // can set clampValue to zero (high values = noise) or clampLimit

    if (returnAmplitude) {
      return torch::stack({tau, peak}, 0).squeeze();
    }
    else {
      return tau.squeeze();
    }
  }

  private:

  inline
  torch::Tensor responseGPU(const torch::Tensor& a, const torch::Tensor& lf) {
    torch::Tensor lorentzian = a / torch::sqrt(1 + (fGPU * lf).square());
    return irGPU * lorentzian;
  }

  inline
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
  responseGPUHess(const torch::Tensor& a, const torch::Tensor& lf, const torch::Tensor& objective) {
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> retVal;
    auto& [loss, dda, ddl, d2da2, d2dl2, d2dadl] = retVal; // avoid copy constructor

    // using formulae
    // d/dx (f(x) - b)^2 = 2 f'(x) (f(x) - b) = f'(x) residual
    // d^2/dx^2 (f(x) - b)^2 = 2 ((f(x) - b) f''(x) + f'(x)^2) = residual f''(x) + f'(x)^2
    // d/dx d/dy (f(x, y) - b)^2 = 2 ((f - b) fxy + fy fx) = residual fxy + fy fx
    // note we omit common factor of 2

    torch::Tensor memoOmega2 = (fGPU * lf).square();

    torch::Tensor lorentzianSq = 1 / (1 + memoOmega2);
    torch::Tensor lorentzian = torch::sqrt(lorentzianSq);

    torch::Tensor f_a = irGPU * lorentzian;
    torch::Tensor outputs = a * f_a;
    torch::Tensor residuals = outputs - objective;

    loss = torch::mean(residuals.square(), -1, true);

    dda = torch::mean(f_a * residuals, -1, true);

    torch::Tensor f_aa = f_a.square();
    d2da2 = torch::mean(f_aa, -1, true);

    torch::Tensor f_l_ = -fGPU.square() * lf * lorentzianSq;
    torch::Tensor f_l = f_l_ * outputs;
    ddl = torch::mean(f_l * residuals, -1, true);

    torch::Tensor f_ll = f_l * lorentzianSq * (1 - 2 * memoOmega2);
    d2dl2 = torch::mean(f_ll * residuals + f_l.square(), -1, true);

    torch::Tensor f_la = f_l_ * f_a;
    d2dadl = torch::mean(f_la * residuals + f_l * f_a, -1, true);

    return retVal;
  }

  uint nFreqs;
  float lambda0;
  float decreaseFactor;
  float increaseFactor;
  float clampLimit;
  float clampValue;
  uint baselineSubtractInd;
  uint startFreqInd;

  torch::TensorOptions options;

  torch::Tensor irGPU;
  torch::Tensor fGPU;

  torch::Tensor regressor;
};

/////////////////////////////////////////////////////////////////////////////////////////////

class fitComplex {
  public:
  fitComplex(const torch::Tensor& _irGPU, const torch::Tensor& _angFreqGPU,
               float _lambda0, float _decreaseFactor, float _increaseFactor,
               float _clampLimit, float _clampValue, uint _baselineSubtractInd, uint _startFreqInd) {

    lambda0 = _lambda0;
    decreaseFactor = _decreaseFactor;
    increaseFactor = _increaseFactor;
    clampLimit = _clampLimit;
    clampValue = _clampValue;
    baselineSubtractInd = _baselineSubtractInd;
    startFreqInd = _startFreqInd;

    nFreqs = _irGPU.size(-1);
    if (_angFreqGPU.size(-1) != nFreqs) throw std::invalid_argument("Impulse response and frequency axis have different lengths.");

    irGPU = _irGPU;
    fGPU = _angFreqGPU;

    options = torch::TensorOptions()
              .dtype(_angFreqGPU.dtype())
              .device(_irGPU.device())
              .layout(torch::kStrided)
              .requires_grad(false);

    torch::Tensor sampleLifeTimes = torch::linspace(0.5, 5, 10, options).unsqueeze(-1);
    torch::Tensor selfCalibrations = responseGPU(torch::tensor(1, options), sampleLifeTimes, torch::tensor(0, options));
    selfCalibrations /= selfCalibrations.index({Ellipsis, Slice(0, 1)});
    selfCalibrations = torch::sum(torch::view_as_real(selfCalibrations).square_(), -1, false);
    torch::Tensor integrals = torch::sum(selfCalibrations, -1, false).squeeze();
    torch::Tensor moments = torch::sum(selfCalibrations * fGPU, -1, false).squeeze();
    moments /= integrals;

    // fit to the profile TODO reimplement for multi channel IRF
    torch::Tensor independent = torch::stack({torch::ones_like(moments, options),
                                              moments,
                                              moments.square(),
                                              moments.pow(3),
                                              }, -1);
    regressor = torch::matmul(torch::matmul(torch::linalg_inv(torch::matmul(independent.t(), independent)), independent.t()), sampleLifeTimes);
  }

  torch::Tensor run(torch::Tensor& traces, uint nOptSteps, bool returnAmplitude, bool inFreqDomain=false) {
    torch::Tensor objective;
    if (inFreqDomain) {
      if (traces.dtype() != irGPU.dtype() || traces.size(-1) != nFreqs) throw  std::invalid_argument("Frequency Domain traces not of right type and/or size.");
      objective = traces;
    }
    else {
      if (baselineSubtractInd)
        traces -= torch::mean(traces.index({Ellipsis, Slice(0, baselineSubtractInd)}), -1, true);
      objective = torch::fft_rfft(traces, std::nullopt, -1, "backward").index({Ellipsis, Slice(startFreqInd, startFreqInd+nFreqs)});

      if (startFreqInd == 0) {
        objective.index({Ellipsis, 0}) = torch::abs(torch::real(objective.index({Ellipsis, 0}))); // if the DC is spuriously negative TODO better than this
      }
    }

    torch::Tensor objectiveInit = torch::sum(torch::view_as_real(objective).square(), -1, false);
    torch::Tensor peak = torch::real(objective.index({Ellipsis, 0})).unsqueeze(-1).clone();
    torch::Tensor integrals = torch::sum(objectiveInit, -1, true);
    torch::Tensor moments = torch::sum(objectiveInit * fGPU, -1, true);
    moments /= integrals;
    torch::Tensor tau = regressor.index({0}) + moments * (regressor.index({1}) + moments * (regressor.index({2}) + moments * regressor.index({3})));
    torch::clip_out(tau, tau, 0.1f, 5.5f);

    torch::Tensor dt = torch::zeros(peak.sizes(), options);

    torch::Tensor lambdai = torch::full(peak.sizes(), lambda0, options);

    for (uint its = 0; its < nOptSteps; ++its) {
      auto response = responseGPUHess(peak, tau, dt, objective);
      auto& [errorVec, gradientPK, gradientLF, hessianPK, hessianLF, hessianPKLF, gradientDT, hessianDT, hessianLFDT] = response;

      torch::Tensor lambdaDampening = (1 + lambdai);
      hessianPK += (-1.1f * hessianPK) * (hessianPK < 0) + 1e-7f * (hessianPK == 0);
      hessianPK *= lambdaDampening;
      hessianLF += (-1.1f * hessianLF) * (hessianLF < 0) + 1e-7f * (hessianLF == 0);
      hessianLF *= lambdaDampening;
      hessianDT += (-1.1f * hessianDT) * (hessianDT < 0) + 1e-7f * (hessianDT == 0);
      hessianDT *= lambdaDampening;

      torch::Tensor hessianNorm = hessianPK * hessianLF;
      torch::Tensor hessianPKLFSq = hessianPKLF.square();
      hessianNorm -= hessianPKLFSq * (hessianNorm != hessianPKLFSq);
      hessianNorm = -1 / hessianNorm;

      torch::Tensor stepDeltaPK = (gradientPK * hessianLF - gradientLF * hessianPKLF) * hessianNorm;
      torch::Tensor stepDeltaLF = (gradientLF * hessianPK - gradientPK * hessianPKLF) * hessianNorm;

      hessianNorm = hessianDT * hessianLF; // for simplicity do semi-independently
      torch::Tensor hessianLFDTSq = hessianLFDT.square();
      hessianNorm -= hessianLFDTSq * (hessianNorm != hessianLFDTSq);
      torch::Tensor stepDeltaDT = (gradientLF * hessianLFDT - gradientDT * hessianLF) / hessianNorm;

      torch::nan_to_num_out(stepDeltaDT, stepDeltaDT, 0); // this variable has a danger of having zero 2nd deriv and giving nan

      torch::Tensor _peak = peak + stepDeltaPK;
      torch::Tensor _tau = tau + stepDeltaLF;
      torch::Tensor _dt = dt + stepDeltaDT;

      torch::Tensor fitVec = responseGPU(_peak, _tau, _dt);
      fitVec -= objective;
      fitVec = torch::view_as_real(fitVec);
      fitVec.square_();
      torch::Tensor errorVec2 = torch::sum(fitVec, {-2, -1}, true).squeeze(-1);
      errorVec2 *= 1. / nFreqs;

      torch::Tensor errorImprovement = errorVec - errorVec2;

      torch::Tensor improvedMask = (errorImprovement > 0);
      peak += improvedMask * stepDeltaPK;
      tau += improvedMask * stepDeltaLF;
      dt += improvedMask * stepDeltaDT;


      lambdai *= decreaseFactor * improvedMask + increaseFactor * ~improvedMask;
      torch::clamp_out(lambdai, lambdai, 1e-7f, 1e7f);
    }

    tau.abs_();
    peak.abs_();
    peak = torch::where(tau < clampLimit, peak, 0); // assume that these are misinterpreted DC traces
    tau  = torch::where(tau < clampLimit, tau, clampValue); // can set clampValue to zero (high values = noise) or clampLimit

    if(returnAmplitude) {
      return torch::stack({tau, peak * tau}, 0).squeeze();
    }
    else {
      return tau.squeeze();
    }
  }

  private:

  inline
  torch::Tensor responseGPU(const torch::Tensor& a, const torch::Tensor& lf, const torch::Tensor& dt) {
    static const auto i = c10::complex<float>(0.0, 1.0);
    torch::Tensor lorentzian = (a * lf) / (1 + i * (fGPU * lf));
    torch::Tensor shift = torch::exp(i * (dt * fGPU));
    return irGPU * lorentzian * shift;
  }

  inline
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
  responseGPUHess(const torch::Tensor& a, const torch::Tensor& lf, const torch::Tensor& dt, const torch::Tensor& objective) {
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> retVal;
    auto& [loss, dda, ddl, d2da2, d2dl2, d2dadl, dds, d2ds2, d2dsdl] = retVal; // avoid copy constructor

    // using formulae
    // d/dx (f(x) - b)^2 = 2 f'(x) (f(x) - b) = f'(x) residual
    // d^2/dx^2 (f(x) - b)^2 = 2 ((f(x) - b) f''(x) + f'(x)^2) = residual f''(x) + f'(x)^2
    // d/dx d/dy (f(x, y) - b)^2 = 2 ((f - b) fxy + fy fx) = residual fxy + fy fx
    // note we omit common factor of 2

    static const auto i = c10::complex<float>(0.0, 1.0);
    torch::Tensor memoOmega = fGPU * lf;

    torch::Tensor irShift = irGPU * torch::exp(i * (dt * fGPU));
    torch::Tensor expDeriv = i / (1 + i * memoOmega);

    torch::Tensor f_a = irShift * lf / (1 + i * memoOmega);

    torch::Tensor outputs = a * f_a;
    torch::Tensor residuals = outputs - objective;
    residuals.conj_physical_();
    auto& residualsConj = residuals;

    loss = torch::sum(torch::view_as_real(residuals).square(), {-2, -1}, true).squeeze(-1); // equivalent to |residuals|^2
    loss *= 1. / nFreqs;

    dda = torch::sum(torch::real(f_a * residualsConj), -1, true); // equivalent to Re[fa residuals*]
    dda *= 1. / nFreqs;

    d2da2 = torch::sum(torch::view_as_real(f_a).square(), {-2, -1}, true).squeeze(-1);
    d2da2 *= 1. / nFreqs;

    torch::Tensor f_la = - expDeriv.square() * irShift;

    torch::Tensor f_l = a * f_la;
    ddl = torch::sum(torch::real(f_l * residualsConj), -1, true);
    ddl *= 1. / nFreqs;

    torch::Tensor f_ll = -2 * fGPU * f_l * expDeriv;
    d2dl2 = torch::sum(torch::real(f_ll * residualsConj), -1, true); // checkme
    d2dl2 += torch::sum(torch::view_as_real(f_l).square(), {-2, -1}, true).squeeze(-1);
    d2dl2 *= 1. / nFreqs;

    torch::Tensor f_l_conj = torch::conj_physical(f_l);
    d2dadl = torch::sum(torch::real(f_la * residualsConj), -1, true);
    d2dadl += torch::sum(torch::real(f_l_conj * f_a), -1, true);
    d2dadl *= 1. / nFreqs;

    torch::Tensor f_s = (a * fGPU) * i * f_a;
    dds = torch::sum(torch::real(f_s * residualsConj), -1, true);
    dds *= 1. / nFreqs;

    torch::Tensor f_ss = f_s * (i * fGPU);
    d2ds2 = torch::sum(torch::real(f_ss * residuals), -1, true);
    d2ds2 += torch::sum(torch::view_as_real(f_s).square(), {-2, -1}, true).squeeze(-1);
    d2ds2 *= 1. / nFreqs;

    torch::Tensor f_ls = f_l * (i * fGPU);
    d2dsdl = torch::sum(torch::real(f_ls * residualsConj), -1, true);
    d2dsdl += torch::sum(torch::real(f_l_conj * f_s), -1, true);
    d2dsdl *= 1. / nFreqs;

    return retVal;
  }

  uint nFreqs;
  float lambda0;
  float decreaseFactor;
  float increaseFactor;
  float clampLimit;
  float clampValue;
  uint baselineSubtractInd;
  uint startFreqInd;

  torch::TensorOptions options;

  torch::Tensor irGPU;
  torch::Tensor fGPU;

  torch::Tensor derivative;
  torch::Tensor regressor;
};


/////////////////////////////////////////////////////////////////////////////////////////////


PYBIND11_MODULE(lmfit, m) {

  namespace py = pybind11;
  using namespace pybind11::literals;

  m.doc() = "Module for least squares fiting of lifetime decay traces.";

  py::class_<fit> fit(m, "fit", "Least squares fit in the frequency domain.");

  fit.def(py::init<const torch::Tensor&, const torch::Tensor&,
                   float, float, float, float, float, uint, uint>(),
          "Instantiates a fitter, which can then be called with .run().\n"
          "Arguments:\n"
          "  irGPU: Pytorch Tensor, representing the impulse response function\n"
          "  angFreqGPU: Pytorch Tensor, representing the angular frequencies corresponding to the impulse response\n"
          "  lambda0, decreaseFactor, increaseFactor: Levenberg-Marquardt initial damping parameter, and the change after each (un)successful iteration\n"
          "  clampLimit, clampValue: clamp the lifetime at a given limit, and the replacement value (eg either clampValue=clampLimit or clampValue=0, for high lifetime results due to pure noise)\n"
          "  baselineSubtractInd: if nonzero, subtracts the mean of these first indices from the trace, to remove offset / DC bias, which affect the fit\n"
          "  startFreqInd: if nonzero, fits by skipping the first given frequency components (irGPU, angFreqGPU must reflect this)\n",
          "irGPU"_a, "angFreqGPU"_a,
          "lambda0"_a=1, "decreaseFactor"_a=1/2, "increaseFactor"_a=3,
          "clampLimit"_a=6.5, "clampValue"_a=5.0, "baselineSubtractInd"_a=0, "startFreqInd"_a=0);

  fit.def("run", &fit::run,
          "Run least squares fit in the frequency domain.\n"
          "Arguments:\n"
          "  traces: Pytorch Tensor representing the traces\n"
          "  nIterations: number of optimization steps to perform on the batch\n"
          "  returnAmplitude: if true, returns a fit to intensity as well as the lifetime; lifetime and amplitude are stacked as Tensors, eg can be split as:\n"
          "                   lifetime, intensity = fit.run(traces, [nIterations,] returnAmplitude=True)\n",
          "traces"_a, "nIterations"_a=50, "returnAmplitude"_a=false, "inFreqDomain"_a=false);



  py::class_<fitComplex> fitComplex(m, "fitComplex", "Least squares fit in the frequency domain.");

  fitComplex.def(py::init<const torch::Tensor&, const torch::Tensor&,
                          float, float, float, float, float, uint, uint>(),
                 "Instantiates a fitter, which can then be called with .run(). Works in the same manner as `fit` except that it considers phase",
                 "irGPU"_a, "angFreqGPU"_a,
                 "lambda0"_a=1, "decreaseFactor"_a=1/2, "increaseFactor"_a=3,
                 "clampLimit"_a=6.5, "clampValue"_a=5.0, "baselineSubtractInd"_a=0, "startFreqInd"_a=0);

  fitComplex.def("run", &fitComplex::run, "Run weighted least squares fit in the complex frequency domain.",
                 "traces"_a, "nIterations"_a=50, "returnAmplitude"_a=false, "inFreqDomain"_a=false);

}
#undef NLMMODULE