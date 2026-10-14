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
# - Per-channel rms, with unusable channels set to NaN
# -----------------------------------------------------------------------------
def _channel_rms(data, rms):
    nchans = data.shape[0]

    if rms is None:
        print(
            "Calculating RMS from data. If you want to set it manually, "
            "assign a value to 'self.rms' before running the sampler."
        )

        rms = scipy.stats.median_abs_deviation(
            data, scale="normal", axis=(-2, -1), nan_policy="omit"
        )

    try:
        rms = np.broadcast_to(rms, (nchans,)).astype(np.float64)
    except ValueError:
        raise ValueError(
            f"RMS has shape {np.shape(rms)}, but it should be either a scalar or have shape ({nchans},)."
        )

    # A zero or non-finite rms would make the likelihood non-finite
    bad = ~(np.isfinite(rms) & (rms > 0.00))
    if np.all(bad):
        raise ValueError("No channel has a valid RMS: the likelihood cannot be computed.")
    if np.any(bad):
        print(f"Warning. {bad.sum()} of {nchans} channels have no valid RMS and are left out of the likelihood.")
        rms[bad] = np.nan

    return rms


# -----------------------------------------------------------------------------
# - Gaussian likelihood without correlation
# -----------------------------------------------------------------------------
class Normal:
    def __init__(self, rms=None):
        self.rms = rms

    def _build(self, data, **kwargs):
        self.rms = _channel_rms(data, self.rms)
 
        # Blanked voxels do not enter the likelihood
        nvalid = np.isfinite(data).sum(axis=(-2, -1))

        return {"rms": self.rms, "nvalid": nvalid}

    def _compute(self, theta):
        """Likelihood function for Gaussian noise without correlation."""
        
        model_out = self._get_model(theta, convolve=True)
        if not isinstance(model_out, tuple):
            return -np.inf
        model, data, _ = model_out

        log_like = np.abs(data - model) / self.rms[:, None, None]
        log_like = -0.50 * np.nansum(log_like ** 2)

        log_norm = np.log(2.00 * np.pi * self.rms**2)
        log_norm = -0.50 * np.nansum(log_norm * self.nvalid)

        regularization = self._regularize_vdisp(theta, alpha=1.00)
        
        return log_like + log_norm + regularization


# -----------------------------------------------------------------------------
# - Radio-interferometric likelihood
# -----------------------------------------------------------------------------
class NormalRI:
    def __init__(self, rms=None, mode="image", threshold="auto"):
        if mode not in ["image", "fourier", "full"]:
            raise ValueError("Invalid value for 'mode' argument. Please use 'image', 'fourier' or 'full'.")
        if threshold is None or (isinstance(threshold, str) and threshold != "auto"):
            raise ValueError("Invalid value for 'threshold' argument. Please use 'auto' or a number.")
        self.rms = rms
        self.mode = mode
        self.threshold = threshold

    def _build(self, data, beam=None, **kwargs):
        self.rms = _channel_rms(data, self.rms)

        out = {"rms": self.rms}

        if beam is None:
            raise ValueError(f"The '{self.mode}' mode of 'NormalRI' requires a beam kernel.")

        if self.mode != "image":

            shape = _padded_shape(data, beam)
            beam_fft = np.fft.rfft2(_kernel_at_origin(beam, shape), axes=(-2, -1)).real
            data_fft = np.fft.rfft2(np.nan_to_num(data), s=shape, axes=(-2, -1))

            weight = np.nan_to_num(1.00 / self.rms**2)[:, None, None]
            weight = weight * _rfft_pairs(shape[1]) / (shape[0] * shape[1])

            out["fft_shape"] = shape

            if self.mode == "fourier":
                out["fft_quad"] = weight * beam_fft
                out["fft_cross"] = weight * np.conj(data_fft)
            else:
                beam_max = beam_fft.max(axis=(-2, -1), keepdims=True)
                threshold = self.threshold

                # Ringing level from the negative cells, in units of the maximum
                if threshold == "auto":
                    neg = beam_fft < 0
                    ring = np.sum(np.where(neg, beam_fft, 0.00) ** 2, axis=(-2, -1), keepdims=True)
                    ring = np.sqrt(ring / np.maximum(neg.sum(axis=(-2, -1), keepdims=True), 1))
                    threshold = 3.00 * ring / beam_max

                # Inverse variance of the Fourier cells sampled by the beam
                keep = beam_fft > threshold * beam_max
                icov = np.where(keep, weight / np.where(keep, beam_fft, 1.00), 0.00)

                out["fft_threshold"] = threshold
                out["fft_keep"] = keep
                out["data_fft"] = data_fft
                out["beam_fft"] = beam_fft
                out["icov"] = icov

        return out

    def _compute(self, theta):
        """Likelihood function for radio-interferometric measurements"""
        
        model_out = self._get_model(theta, convolve=False)
        if not isinstance(model_out, tuple):
            return -np.inf
        model_clean, data, _ = model_out

        if self._likelihood.mode == "image":
            model_dirty = self._smooth_model(model_clean)

            log_like = model_clean * (model_dirty - 2.00 * data)
            log_like = -0.50 * np.nansum(log_like / self.rms[:, None, None]**2)
        else:
            model_fft = np.fft.rfft2(np.nan_to_num(model_clean), s=self.fft_shape, axes=(-2, -1))

            if self._likelihood.mode == "fourier":
                log_like = np.sum(self.fft_quad * (model_fft.real**2 + model_fft.imag**2))
                log_like = -0.50 * (log_like - 2.00 * np.sum((self.fft_cross * model_fft).real))
            else:
                resid = self.data_fft - self.beam_fft * model_fft
                log_like = -0.50 * np.sum(self.icov * (resid.real**2 + resid.imag**2))

        log_norm = 0.00

        regularization = self._regularize_vdisp(theta, alpha=1.00)
        
        return log_like + log_norm + regularization
