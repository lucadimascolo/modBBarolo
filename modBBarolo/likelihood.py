import re

import numpy as np
import scipy.fft
import scipy.stats

# -----------------------------------------------------------------------------
# - Kernel on a grid of the given shape, with its peak at the origin
# -----------------------------------------------------------------------------
def _kernel_at_origin(kernel, shape):
    kernel = kernel if kernel.ndim == 3 else kernel[None]
    ny, nx = shape

    grid = np.zeros((kernel.shape[0], ny, nx))
    for c in range(kernel.shape[0]):
        y0, x0 = np.unravel_index(np.argmax(kernel[c]), kernel[c].shape)
        iy = np.arange(kernel.shape[1]) - y0
        ix = np.arange(kernel.shape[2]) - x0
        sy = (iy >= -(ny // 2)) & (iy < ny - ny // 2)
        sx = (ix >= -(nx // 2)) & (ix < nx - nx // 2)
        grid[c][np.ix_(iy[sy] % ny, ix[sx] % nx)] = kernel[c][np.ix_(sy, sx)]
    return grid


# -----------------------------------------------------------------------------
# - Multiplicity of the rfft2 half-plane
# -----------------------------------------------------------------------------
def _rfft_pairs(nx):
    pair = np.full(nx // 2 + 1, 2.00)
    pair[0] = 1.00
    if nx % 2 == 0:
        pair[-1] = 1.00
    return pair


# -----------------------------------------------------------------------------
# - Zero-padded grid of the image-space convolution
# -----------------------------------------------------------------------------
def _padded_shape(data, beam):
    return tuple(
        scipy.fft.next_fast_len(int(i + k - 1), real=True)
        for i, k in zip(data.shape[-2:], beam.shape[-2:])
    )


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
    def __init__(self, rms=None, mode="full"):
        if mode not in ["full", "fast"]:
            raise ValueError("Invalid value for 'mode' argument. Please use 'full' or 'fast'.")
        self.rms = rms
        self.mode = mode

    def _build(self, data, beam=None, **kwargs):
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

        out = {"rms": self.rms}

        if self.mode == "fast":
            if beam is None:
                raise ValueError("The 'fast' mode of 'NormalRI' requires a beam kernel.")

            shape = _padded_shape(data, beam)
            beam_fft = np.fft.rfft2(_kernel_at_origin(beam, shape), axes=(-2, -1)).real

            weight = np.nan_to_num(1.00 / self.rms**2)[:, None, None]
            out["ri_shape"] = shape
            out["ri_quad"] = weight * _rfft_pairs(shape[1]) * beam_fft / (shape[0] * shape[1])
            out["ri_data"] = weight * np.nan_to_num(data)

        return out

    def _compute(self, theta):
        """Likelihood function for radio-interferometric measurements"""
        
        model_out = self._get_model(theta, convolve=False)
        if not isinstance(model_out, tuple):
            return -np.inf
        model_clean, data, _ = model_out

        if self._likelihood.mode == "fast":
            # sum(m * (B * m)) from the model power, without inverse FFT
            model_clean = np.nan_to_num(model_clean)
            model_fft = np.fft.rfft2(model_clean, s=self.ri_shape, axes=(-2, -1))
            log_like = np.sum(self.ri_quad * (model_fft.real**2 + model_fft.imag**2))
            log_like = -0.50 * (log_like - 2.00 * np.sum(model_clean * self.ri_data))
        else:
            model_dirty = self._smooth_model(model_clean)

            log_like = model_clean * (model_dirty - 2.00 * data)
            log_like = -0.50 * np.nansum(log_like / self.rms[:, None, None]**2)

        log_norm = 0.00

        regularization = self._regularize_vdisp(theta, alpha=1.00)
        
        return log_like + log_norm + regularization


# -----------------------------------------------------------------------------
# - Radio-interferometric likelihood per Fourier cell
# -----------------------------------------------------------------------------
class NormalFourier(NormalRI):
    def __init__(self, rms=None, filter=None):
        super().__init__(rms=rms)
        self.filter = filter

    def _build(self, data, beam=None, **kwargs):
        out = super()._build(data, **kwargs)

        if beam is None:
            raise ValueError("The 'NormalFourier' likelihood requires a beam kernel.")

        shape = _padded_shape(data, beam)
        cells = (shape[0], shape[1] // 2 + 1)
        beam_fft = np.fft.rfft2(_kernel_at_origin(beam, shape), axes=(-2, -1)).real
        data_fft = np.fft.rfft2(np.nan_to_num(data), s=shape, axes=(-2, -1))

        # Spatial frequencies in cycles per pixel
        ky, kx = np.meshgrid(np.fft.fftfreq(shape[0]), np.fft.rfftfreq(shape[1]), indexing="ij")

        if self.filter is None:
            filt = np.ones(cells)
        elif callable(self.filter):
            filt = np.asarray(self.filter(ky, kx), dtype=float)
        else:
            filt = np.asarray(self.filter, dtype=float)

        if filt.shape not in [cells, (data.shape[0],) + cells]:
            raise ValueError(
                f"Filter has shape {filt.shape}, but it should be {cells} or {(data.shape[0],) + cells}."
            )

        weight = np.nan_to_num(1.00 / self.rms**2)[:, None, None]
        weight = weight * filt * _rfft_pairs(shape[1]) / (shape[0] * shape[1])

        out["rf_shape"] = shape
        out["rf_freq"] = (ky, kx)
        out["rf_filter"] = filt
        out["rf_quad"] = weight * beam_fft
        out["rf_cross"] = weight * np.conj(data_fft)
        return out

    def _compute(self, theta):
        """Likelihood function for radio-interferometric measurements per Fourier cell"""

        model_out = self._get_model(theta, convolve=False)
        if not isinstance(model_out, tuple):
            return -np.inf
        model, _, _ = model_out

        model_fft = np.fft.rfft2(np.nan_to_num(model), s=self.rf_shape, axes=(-2, -1))

        log_like = np.sum(self.rf_quad * (model_fft.real**2 + model_fft.imag**2))
        log_like = -0.50 * (log_like - 2.00 * np.sum((self.rf_cross * model_fft).real))

        regularization = self._regularize_vdisp(theta, alpha=1.00)

        return log_like + regularization
