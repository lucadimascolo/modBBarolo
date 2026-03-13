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
        
        model, data, _ = self._get_model(theta, convolve=True)

        log_like = np.abs(data - model) / self.rms[:, None, None] 
        log_like = -0.50 * np.nansum(log_like ** 2)

        log_norm = np.log(2.00 * np.pi * self.rms**2)
        log_norm = -0.50 * np.nansum(log_norm * data.shape[1] * data.shape[2])

        return log_like + log_norm


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
        
        model_clean, data, _ = self._get_model(theta, convolve=False)
        model_dirty = self._smooth_model(model_clean)

        log_like = model_clean * (model_dirty - 2.00 * data)
        log_like = -0.50 * np.nansum(log_like / self.rms[:, None, None]**2)

        log_norm = 0.00

        return log_like + log_norm