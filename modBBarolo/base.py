import matplotlib.pyplot as plt

from pyBBarolo.bayesian import get_distribution
from pyBBarolo.bayesian import libBB, ctypes, reshapePointer
from pyBBarolo.bayesian import Rings
from pyBBarolo.bayesian import BayesianBBarolo

import nautilus
import corner

import inspect
from datetime import datetime

import numpy as np
import scipy.stats
from scipy.signal import fftconvolve
from astropy.io import fits as pyfits

from .likelihood import LL_Normal

import multiprocess
multiprocess.set_start_method("fork")

is_positive = ("vrot", "vdisp", "inc", "xpos", "ypos", "dens", "z0", "radmax", "norm", "rdisk")

_active_sampler = None

# -----------------------------------------------------------------------------
# Initialize BBarolo
# -----------------------------------------------------------------------------
class Init(BayesianBBarolo):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._vrot_r0_removed_idx = None
        self._full_freepar_idx = None

    def initialize(self, rmax, rnum, init={}, set_options={}, add_zero=True):
        rmin = rmax / (1.00 + 2.00 * (rnum - 1.00))

        self.radii = np.linspace(rmin, rmax, rnum)
        self._add_zero = add_zero
        if add_zero:
            self.radii = np.append(0.00, self.radii)

        self.init(radii=self.radii,**init)

        self.set_options(**set_options)

    def _setup(self, freepar, **kwargs):
        super()._setup(freepar, **kwargs)

        if self._add_zero and 'vrot' in self.freepar_idx and len(self.freepar_idx['vrot']) > 1:
            self._full_freepar_idx = {k: v.copy() for k, v in self.freepar_idx.items()}

            removed_idx = self.freepar_idx['vrot'][0]
            self._vrot_r0_removed_idx = removed_idx

            self.freepar_idx['vrot'] = self.freepar_idx['vrot'][1:]

            for key in self.freepar_idx:
                self.freepar_idx[key] = np.where(
                    self.freepar_idx[key] > removed_idx,
                    self.freepar_idx[key] - 1,
                    self.freepar_idx[key]
                )

            self.freepar_names.remove('vrot1')
            self.ndim -= 1

    def _update_rings(self, rings, theta):
        if self._vrot_r0_removed_idx is not None:
            theta = np.insert(theta, self._vrot_r0_removed_idx, 0.0)
            saved_idx = self.freepar_idx
            self.freepar_idx = self._full_freepar_idx
            result = super()._update_rings(rings, theta)
            self.freepar_idx = saved_idx
            return result
        else:
            return super()._update_rings(rings, theta)


# -----------------------------------------------------------------------------
# Multiprocessing handlers
# -----------------------------------------------------------------------------
def _mp_log_likelihood(theta):
    return _active_sampler._log_likelihood(theta)

def _mp_prior_transform(u):
    return _active_sampler._prior_transform(u)


# -----------------------------------------------------------------------------
# Sampler class
# -----------------------------------------------------------------------------
class Sampler:
    _log_likelihood = LL_Normal
    
    def __init__(self, bbobj, free_params, method_norm="constant", output=None):
        self.bbobj = bbobj
        self.free_params = free_params
        self.method_norm = method_norm
        self.output = output

        if self.output is None:
            self.output = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.bbobj._opts.add_params(sm=False)

        self.bbobj._setup(self.free_params, useBBres=False)

        if self.bbobj.mask is None:
            self.bbobj.mask = np.ones_like(self.bbobj.data, dtype=bool)

        if not self.bbobj.useNorm:
            raise ValueError("modBBarolo does not support setting 'dens' as a free parameter.")
        
        self.freepar_names = list(self.bbobj.freepar_names)
        self.freepar_idx = dict(self.bbobj.freepar_idx)
        self.prior_distr = dict(self.bbobj.prior_distr)

        if self.method_norm == "constant":
            self.freepar_names.append("norm")
            self.freepar_idx["norm"] = len(self.freepar_names) - 1

            distr_kwargs = {key: value for key, value in self.bbobj.priors["norm"].items() if key != "name"}

            self.prior_distr["norm"] = get_distribution(
                self.bbobj.priors["norm"]["name"], **distr_kwargs
            )

        elif self.method_norm == "exponential":
            hdr = pyfits.getheader(self.bbobj.inp.fname)
            self.pixscale = abs(hdr['CDELT2']) * 3600
            del hdr

            for pname in ("norm", "rdisk"):
                self.freepar_names.append(pname)
                self.freepar_idx[pname] = len(self.freepar_names) - 1

                distr_kwargs = {key: value for key, value in self.bbobj.priors[pname].items() if key != "name"}
                self.prior_distr[pname] = get_distribution(
                    self.bbobj.priors[pname]["name"], **distr_kwargs
                )

        self.rms = None

        self._beam_kernel = self._build_beam_kernel()

# -----------------------------------------------------------------------------
# - Beam convolution (pure Python, matching BBarolo's setCfield)
# -----------------------------------------------------------------------------
    def _build_beam_kernel(self):
        """Build 2D Gaussian beam kernel matching BBarolo's cfield construction."""
        hdr = pyfits.getheader(self.bbobj.inp.fname)

        bmaj = hdr['BMAJ'] * 3600  # degrees -> arcsec
        bmin = hdr['BMIN'] * 3600  # degrees -> arcsec
        bpa  = hdr.get('BPA', 0)   # degrees

        pixsizeX = abs(hdr['CDELT1']) * 3600  # degrees -> arcsec
        pixsizeY = abs(hdr['CDELT2']) * 3600  # degrees -> arcsec

        # BBarolo convention: convolving from point beam (0,0,0) to target beam
        # gives Con = {bmaj, bmin, bpa-90}
        phi = np.radians(bpa - 90.0)
        cs, sn = np.cos(phi), np.sin(phi)

        # Kernel extent (same factor as BBarolo: sqrt(-log(1e-4)/log(2)))
        extend = np.sqrt(-np.log(1e-4) / np.log(2.0))
        xr_ext = 0.5 * bmaj * extend
        yr_ext = 0.5 * bmin * extend

        x_max = max(abs(xr_ext * cs), abs(yr_ext * sn))
        y_max = max(abs(xr_ext * sn), abs(yr_ext * cs))

        Xmax = round(x_max / pixsizeX)
        Ymax = round(y_max / pixsizeY)

        # Build kernel on pixel grid
        ii, jj = np.meshgrid(np.arange(-Xmax, Xmax + 1),
                             np.arange(-Ymax, Ymax + 1))

        x = ii * pixsizeX  # pixel -> arcsec
        y = jj * pixsizeY

        xr =  x * cs + y * sn
        yr = -x * sn + y * cs

        argfac = -4.0 * np.log(2.0)
        argX = (xr / bmaj) if bmaj != 0 else np.zeros_like(xr)
        argY = (yr / bmin) if bmin != 0 else np.zeros_like(yr)
        kernel = np.exp(argfac * (argX**2 + argY**2))
        kernel[kernel < 1e-4] = 0
        return kernel / kernel.sum()

    def _smooth_model(self, model):
        """Convolve each channel with the beam kernel (pure Python)."""
        # smoothed = np.zeros_like(model, dtype=np.float64)
        smoothed = fftconvolve(model, self._beam_kernel[None, :, :], mode="same", axes=(-2, -1))
        # for z in range(model.shape[0]):
        # #   sm_ = np.fft.rfft2(model[z]) * self._beam_kernel
        # #   smoothed[z] = np.fft.irfft2(sm_)
        #     smoothed[z] = fftconvolve(model[z], self._beam_kernel, mode='same')
        smoothed[np.abs(smoothed) < 1e-12] = 0
        return smoothed

# -----------------------------------------------------------------------------
# - Model normalization
# -----------------------------------------------------------------------------
    def _normalize_model(self, model, data, **kwargs):
        if self.method_norm == "model":
            return model * np.nansum(data) / np.nansum(model)
        elif self.method_norm == "constant":
            return kwargs["norm"] * model
        elif self.method_norm == "exponential":
            norm  = kwargs["norm"]
            rdisk = kwargs["rdisk"]
            rings = kwargs["rings"]
            bhi, blo = kwargs["bhi"], kwargs["blo"]

            x0  = np.mean(rings.r['xpos'])
            y0  = np.mean(rings.r['ypos'])
            inc = np.radians(np.mean(rings.r['inc']))
            phi = np.radians(np.mean(rings.r['phi']))

            x = np.arange(blo[0], bhi[0])
            y = np.arange(blo[1], bhi[1])
            xx, yy = np.meshgrid(x, y)

            xr =  -(xx - x0) * np.sin(phi) + (yy - y0) * np.cos(phi)
            yr = (-(xx - x0) * np.cos(phi) - (yy - y0) * np.sin(phi)) / np.cos(inc)
            R = np.sqrt(xr**2 + yr**2) * self.pixscale

            return  model * norm * np.exp(-R / rdisk)


# -----------------------------------------------------------------------------
# - Prior Transform for nested sampler
# -----------------------------------------------------------------------------
    def _prior_transform(self, u):
        p = np.zeros_like(u)
        for key in self.freepar_idx.keys():
            p[self.freepar_idx[key]] = self.prior_distr[key].ppf(u[self.freepar_idx[key]])
        return p

# -----------------------------------------------------------------------------
# - Build BBarolo model and data+mask for likelihood
# ------------------------------------------------------------------------------
    def _get_model(self, theta):
        for k in self.freepar_idx:
            if k.startswith(is_positive) and np.any(theta[self.freepar_idx[k]] < 0): 
                return -np.inf

        rings = self.bbobj._update_rings(self.bbobj._inri, theta)

        if self.bbobj.update_prof:
            self.bbobj._update_profile(rings)

        # Calculate the model and the boundaries
        model_, bhi, blo, galmod = self.bbobj._calculate_model(rings)

        # Calculate the residuals
        mask = self.bbobj.mask.copy() # [:, blo[1]:bhi[1], blo[0]:bhi[0]]
        data = self.bbobj.data.copy() # [:, blo[1]:bhi[1], blo[0]:bhi[0]]

        data_ = data[:, blo[1]:bhi[1], blo[0]:bhi[0]].copy()

        kwargs = {}
        if self.method_norm == "constant":
            kwargs["norm"] = theta[self.freepar_idx["norm"]]
        elif self.method_norm == "exponential":
            kwargs["norm"]  = theta[self.freepar_idx["norm"]]
            kwargs["rdisk"] = theta[self.freepar_idx["rdisk"]]
            kwargs["rings"] = rings
            kwargs["bhi"], kwargs["blo"] = bhi, blo

        model_ = self._normalize_model(model_, data_, **kwargs)

        model = np.zeros(data.shape)
        model[:, blo[1]:bhi[1], blo[0]:bhi[0]] = model_.copy()

        # Convolve with the beam after normalization (pure Python to avoid
        # C++ Smooth3D heap operations that corrupt malloc in forked processes)
        model = self._smooth_model(model)

        libBB.Galmod_delete(galmod)

        return model, data, mask
    

# -----------------------------------------------------------------------------
# - Main method to run the sampler
# -----------------------------------------------------------------------------
    def run(self, method="nautilus", checkpoint=False, resume=False, threads=1, **kwargs):
        if self.rms is None:
            print(
                "Calculating RMS from data. If you want to set it manually, "
                  "assign a value to 'self.rms' before running the sampler."
            )

            self.rms = scipy.stats.median_abs_deviation(
                self.bbobj.data,
                scale='normal',
                axis=(-2,-1),
                nan_policy='omit'
            )
            self.rms = np.median(self.rms)

        global _active_sampler
        _active_sampler = self

        nlive = 1000
        for key in ["nlive", "n_live"]:
            nlive = kwargs.pop(key, nlive)

        discard_exploration = kwargs.pop("discard_exploration", True)

        sampler_kwargs = {}
        for key in inspect.signature(nautilus.Sampler).parameters.keys():
            if key not in [
                "loglikelihood",
                "prior_transform",
                "n_dim",
                "n_live",
                "filepath",
                "resume",
                "pool",
            ]:
                if key in kwargs:
                    sampler_kwargs[key] = kwargs.pop(key)

        run_kwargs = {}
        for key in inspect.signature(nautilus.Sampler.run).parameters.keys():
            if key not in ["verbose", "discard_exploration"]:
                if key in kwargs:
                    run_kwargs[key] = kwargs.pop(key)

        if checkpoint:
            filepath = f"{self.output}_checkpoint.h5"
        else:
            filepath = None

        if threads > 1:
            pool = multiprocess.Pool(threads)
        else:
            pool = None

        if method == "nautilus":
            self.sampler = nautilus.Sampler(
                prior=_mp_prior_transform,
                likelihood=_mp_log_likelihood,
                n_live=nlive,
                n_dim=len(self.freepar_names),
                filepath=filepath,
                resume=resume,
                pool=pool,
            )

            self.sampler.run(verbose=True, discard_exploration=discard_exploration, **run_kwargs)

            self.samples, weights, _ = self.sampler.posterior()
            self.weights = np.exp(weights - np.max(weights))

        if pool is not None:
            pool.close()
            pool.join()
            
        self.params = np.array([corner.quantile(s, 0.50, weights=self.weights)[0] for s in self.samples.T])

        self.bbobj.params = self.params[:len(self.bbobj.freepar_names)]
        self.bbobj.modCalculated = True


# -----------------------------------------------------------------------------
# - Save corner plot
# -----------------------------------------------------------------------------
    def save_corner(self,sigma=10.00,**kwargs):
        if sigma is not None:
            edges = np.array([corner.quantile(s, [0.16, 0.50, 0.84], weights=self.weights) for s in self.samples.T])
            edges = np.array([[np.maximum(e[1] - sigma * (e[1]-e[0]), self.samples[:,i].min()), 
                               np.minimum(e[1] + sigma * (e[2]-e[1]), self.samples[:,i].max())] for i, e in enumerate(edges)])
        else: 
            edges = np.array([[s.min(), s.max()] for s in self.samples.T])
        
        show_titles = kwargs.pop("show_titles", True)

        corner_kwargs = {}
        for key in inspect.signature(corner.corner).parameters.keys():
            if key not in ["data", "weights", "labels", "show_titles", "range"]:
                if key in kwargs:
                    corner_kwargs[key] = kwargs.pop(key)
                    
        corner.corner(
            self.samples,
            weights=self.weights,
            labels=self.freepar_names,
            show_titles=show_titles,
            range=edges,
            **corner_kwargs
        )
        
        plt.savefig(f"{self.output}_corner.pdf", format="pdf", dpi=300)
        plt.close()


# -----------------------------------------------------------------------------
# - Save best-fit model and outputs using the best-fit parameters
# -----------------------------------------------------------------------------
    def save_best_model(self,plots=True,**kwargs):
            
        if not self.bbobj.modCalculated:
            print ("Sampler has not been run yet. Please run compute() before running this function.")
            return
        
        # Creating a new Rings object for the outputs
        self.bbobj.outri = Rings(self.bbobj._inri.nr)
        self.bbobj.outri.set_rings_from_dict(self.bbobj._inri.r)

        if not hasattr(self.bbobj, "params"):
            self.bbobj.params = self.params[:len(self.bbobj.freepar_names)]

        # Updating output rings with best parameters from the sampling
        self.bbobj.outri = self.bbobj._update_rings(self.bbobj.outri,self.bbobj.params)
    
        # Setting up the output rings in the Galfit object.
        libBB.Galfit_setOutRings(self.bbobj._galfit,self.bbobj.outri._rings)

        self.bbobj._update_profile(self.bbobj.outri)

        # Deriving the last model
        _, ys, xs = self.bbobj.data.shape
        bhi, blo = (ctypes.c_int * 2)(xs,ys), (ctypes.c_int * 2)(0)
        galmod = libBB.Galfit_getModel(self.bbobj._galfit,self.bbobj.outri._rings,bhi,blo,True)

        kwargs = {}
        if self.method_norm == "constant":
            kwargs["norm"] = self.params[self.freepar_idx["norm"]]
        elif self.method_norm == "exponential":
            kwargs["norm"]  = self.params[self.freepar_idx["norm"]]
            kwargs["rdisk"] = self.params[self.freepar_idx["rdisk"]]
            kwargs["rings"] = self.bbobj.outri
            kwargs["bhi"], kwargs["blo"] = bhi, blo

        # Reshaping the model to the correct 3D shape
        model = reshapePointer(libBB.Galmod_array(galmod),self.bbobj.data.shape)
        # Normalizing and copying it back to the C++ Galmod object

        model = self._normalize_model(model, self.bbobj.data, **kwargs)

        # Convolve with the beam after normalization
        model = self._smooth_model(model)

        # Copy smoothed model into the C++ buffer for output writing
        _cpp_buf = reshapePointer(libBB.Galmod_array(galmod), model.shape)
        _cpp_buf[:] = model

        # Writing all the outputs
        libBB.Galfit_writeOutputs(self.bbobj._galfit,galmod,self.bbobj._ellprof,plots)
    
        plt.close()