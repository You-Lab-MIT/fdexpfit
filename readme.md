Fourier domain exponential decay fitting for real-time lifetime imaging.

This GPU-accelerated routine performs parallelized Levenberg-Marquardt fitting, enabling real-time fitting even at the single-shot-per-pixel speeds.
Specifically, fitting is performed on the Fourier transform of the trace of a digitized decay profile.
This avoids back-and-forth FFTs for convolution with the instruments' impulse response function.
In addition, truncation allows a compressed representation of the trace, comprising the response bandwidth, which greatly speeds up fitting.
Convergence speed also significantly benefits, as the function to fit is simpler in the spectral domain.

The code is based on the work presented [here]().

Both a complex-valued and real-valued fit are provided here.
The latter is insensitive to the absolute timing and jitter as discussed in the article.

This program is implemented with PyTorch for portability, and the C++ compiles into a python module via Pybind11.

Contact Federico Presutti for questions.

