import torch

class TraceFilter:
  """
  A bilateral filter for an image consisting of digitized time traces.

  Works by transforming to the Fourier domain, truncating at the given cutoff frequency,
  and performing a weighted averaged based on distance and trace-difference.
  These weights are based on independent Gaussians over the two distances.

  Instantiate an obect and call `run()` on each image or frame.
  """
  def __init__(self, device, sigmaSpace, sigmaIntensity, nFreqs, kernelSize, squaredRadius):
    """
    device:         PyTorch GPU device
    sigmaSpace:     Gaussian constant for the spatial component of the filter
    sigmaIntensity: Gaussian constant for the trace-space distance of the filter
    nFreqs:         Number of frequency components to keep after FFT of the traces
    kernelSize:     Maximum size of the kernel in pixels (must be odd)
    squaredRadius:  Circular acceptance (squared) radius for the filter
                    (pixels farther from the center than this distance will be omitted)
    """
    self.device = device
    self.intensityFactor = -0.5 / sigmaIntensity
    self.nFreqs = nFreqs
    if kernelSize % 2 != 1:
      return ValueError("Need odd kernelSize for symmetry")
    self.kernelSize = kernelSize
    self.squaredRadius = squaredRadius

    coordsSq = [x**2 + y**2 for x in range(-kernelSize//2, kernelSize//2+1)
                            for y in range(-kernelSize//2, kernelSize//2+1)
                if x**2 + y**2 <= squaredRadius]
    self.spaceKernel = torch.exp(-0.5 / sigmaSpace**2 * torch.tensor(coordsSq))
    self.spaceKernel /= torch.sum(self.spaceKernel)
    self.spaceKernel = self.spaceKernel.to(device)

    self.slices = [slice( (self.kernelSize//2+i) % self.kernelSize,
                        -((self.kernelSize//2-i) % self.kernelSize) if self.kernelSize//2 - i else None)
                   for i in range(self.kernelSize)]


  def run(self, traceImage):
    """
    Takes an image where each pixel represents a 1D time trace,
    and returns a filtered image of the Fourier domain of these traces, truncated and absolute value.
    This can be subsequently fed to the fitting routine.
    """
    trsfm = torch.fft.rfft(traceImage, axis=-1, norm="backward")[..., :self.nFreqs]

    trsfm /= torch.sqrt(torch.abs(trsfm)) # variance regularization for spectrum components individually
    torch.nan_to_num_(trsfm, 0)

    stacked = [trsfm[..., self.slices[y], self.slices[x], :]
                 for x in range(-self.kernelSize//2, self.kernelSize//2+1)
                 for y in range(-self.kernelSize//2, self.kernelSize//2+1)
               if x**2 + y**2 <= self.squaredRadius]

    stacked = torch.stack(stacked, axis=-2)

    distanceIntensity = stacked - trsfm[..., self.slices[0], self.slices[0], torch.newaxis, :]
    distanceIntensity = distanceIntensity.real**2 + distanceIntensity.imag**2
    distanceIntensity = torch.sum(distanceIntensity, axis=-1)

    weightsIntensity = torch.exp(self.intensityFactor * distanceIntensity)

    weights = weightsIntensity * self.spaceKernel
    weights /= torch.sum(weights, axis=-1, keepdim=True)

    output = torch.matmul(weights.unsqueeze(-2).to(dtype=stacked.dtype), stacked).squeeze(-2)
    output = torch.abs(output)

    output **= 2

    return output
