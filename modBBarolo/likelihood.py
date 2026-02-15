import numpy as np

# -----------------------------------------------------------------------------
# - Gaussian likelihood without correlation
# -----------------------------------------------------------------------------
def LL_Normal(self, theta):
    """Likelihood function for the fit."""
    
    model, data, mask = self._get_model(theta)

    model, data = model[mask], data[mask]
    
    log_like = np.nansum(np.abs(data - model) ** 2)
    log_like = -0.50 * log_like / self.rms**2
    log_norm = -0.50 * np.count_nonzero(mask) * np.log(2.00 * np.pi * self.rms**2)
        
    return log_like + log_norm

# -----------------------------------------------------------------------------
# - Radio-interferometric likelihood
# -----------------------------------------------------------------------------
def LL_RI(self):
    pass