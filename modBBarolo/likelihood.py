import re

import numpy as np
import scipy.stats

# -----------------------------------------------------------------------------
# - Gaussian likelihood without correlation
# -----------------------------------------------------------------------------
class Normal:
    def __init__(self, rms=None):
        self.rms = rms

    def _build(self, data, **kwargs):
        nchans = data.shape[0]

        if self.rms is None:
            print(
                "Calculating RMS from data. If you want to set it manually, "
                "assign a value to 'self.rms' before running the sampler."
            )

            self.rms = scipy.stats.median_abs_deviation(
                data, scale="normal", axis=(-2, -1), nan_policy="omit"
            )

        try:
            self.rms = np.broadcast_to(self.rms, (nchans,)).copy()
        except ValueError:
            raise ValueError(
                f"RMS has shape {self.rms.shape}, but it should be either a scalar or have shape ({nchans},)."
            )
 
        return {"rms": self.rms}

    def _compute(self, theta):
        """Likelihood function for Gaussian noise without correlation."""
        
        model_out = self._get_model(theta, convolve=True)
        if not isinstance(model_out, tuple):
            return -np.inf
        model, data, _ = model_out

        log_like = np.abs(data - model) / self.rms[:, None, None]
        log_like = -0.50 * np.nansum(log_like ** 2)

        log_norm = np.log(2.00 * np.pi * self.rms**2)
        log_norm = -0.50 * np.nansum(log_norm * data.shape[1] * data.shape[2])

        regularization = self._regularize_vdisp(theta, alpha=1.00)
        
        return log_like + log_norm + regularization


# -----------------------------------------------------------------------------
# - Radio-interferometric likelihood
# -----------------------------------------------------------------------------
class NormalRI:
    def __init__(self, rms=None):
        self.rms = rms

    def _build(self, data, **kwargs):
        nchans = data.shape[0]

        if self.rms is None:
            print(
                "Calculating RMS from data. If you want to set it manually, "
                "assign a value to 'self.rms' before running the sampler."
            )

            self.rms = scipy.stats.median_abs_deviation(
                data, scale="normal", axis=(-2, -1), nan_policy="omit"
            )

        try:
            self.rms = np.broadcast_to(self.rms, (nchans,)).copy()
        except ValueError:
            raise ValueError(
                f"RMS has shape {self.rms.shape}, but it should be either a scalar or have shape ({nchans},)."
            )
 
        return {"rms": self.rms}

    def _compute(self, theta):
        """Likelihood function for radio-interferometric measurements"""
        
        model_out = self._get_model(theta, convolve=False)
        if not isinstance(model_out, tuple):
            return -np.inf
        model_clean, data, _ = model_out
        model_dirty = self._smooth_model(model_clean)

        log_like = model_clean * (model_dirty - 2.00 * data)
        log_like = -0.50 * np.nansum(log_like / self.rms[:, None, None]**2)

        log_norm = 0.00

        regularization = self._regularize_vdisp(theta, alpha=1.00)
        
        return log_like + log_norm + regularization


# -----------------------------------------------------------------------------
# - Radio-interferometric likelihood in Fourier space
# -----------------------------------------------------------------------------
class NormalFourier(NormalRI):
    def __init__(self, rms=None, threshold=1.00E-02):
        super().__init__(rms=rms)
        self.threshold = threshold

    def _build(self, data, beam=None, **kwargs):
        out = super()._build(data, **kwargs)

        if beam is None:
            raise ValueError("The 'NormalFourier' likelihood requires a beam kernel.")

        nchans, ny, nx = data.shape
        kernel = np.broadcast_to(beam, (nchans,) + beam.shape[-2:])

        # Kernel on the image grid, with its peak at the origin
        grid = np.zeros((nchans, ny, nx))
        for c in range(nchans):
            y0, x0 = np.unravel_index(np.argmax(kernel[c]), kernel[c].shape)
            iy = np.arange(kernel.shape[1]) - y0
            ix = np.arange(kernel.shape[2]) - x0
            sy = (iy >= -(ny // 2)) & (iy < ny - ny // 2)
            sx = (ix >= -(nx // 2)) & (ix < nx - nx // 2)
            grid[c][np.ix_(iy[sy] % ny, ix[sx] % nx)] = kernel[c][np.ix_(sy, sx)]

        beam_fft = np.fft.rfft2(grid, axes=(-2, -1)).real

        # Multiplicity of the rfft2 half-plane
        pair = np.full(nx // 2 + 1, 2.00)
        pair[0] = 1.00
        if nx % 2 == 0:
            pair[-1] = 1.00

        # Inverse variance of the Fourier pixels sampled by the beam
        keep = beam_fft > self.threshold * beam_fft.max(axis=(-2, -1), keepdims=True)
        icov = pair / (ny * nx * self.rms[:, None, None] ** 2 * np.where(keep, beam_fft, 1.00))
        icov = np.where(keep, icov, 0.00)

        out["data_fft"] = np.fft.rfft2(np.nan_to_num(data), axes=(-2, -1))
        out["beam_fft"] = beam_fft
        out["icov"] = icov
        return out

    def _compute(self, theta):
        """Likelihood function for radio-interferometric measurements in Fourier space"""

        model_out = self._get_model(theta, convolve=False)
        if not isinstance(model_out, tuple):
            return -np.inf
        model, _, _ = model_out

        model_fft = np.fft.rfft2(np.nan_to_num(model), axes=(-2, -1))
        resid = self.data_fft - self.beam_fft * model_fft

        log_like = -0.50 * np.sum(self.icov * (resid.real ** 2 + resid.imag ** 2))

        regularization = self._regularize_vdisp(theta, alpha=1.00)

        return log_like + regularization
