import matplotlib.pyplot as plt

from pyBBarolo.bayesian import get_distribution
from pyBBarolo.bayesian import libBB, ctypes, reshapePointer
from pyBBarolo.bayesian import BayesianBBarolo

import nautilus
import corner

import inspect
import types
from datetime import datetime

import numpy as np
from astropy.io import fits

from .likelihood import Normal

import multiprocess

multiprocess.set_start_method("fork")

is_positive = (
    "vrot",
    "vdisp",
    "inc",
    "xpos",
    "ypos",
    "dens",
    "z0",
    "radmax",
    "norm",
    "rdisk",
)

_active_sampler = None


# -----------------------------------------------------------------------------
# Initialize BBarolo
# -----------------------------------------------------------------------------
class Init(BayesianBBarolo):
    def __init__(self, fitsname, beam=None, normalize_beam=None, *args, **kwargs):
        super().__init__(fitsname, *args, **kwargs)

        self._beam_kernel = None
        self._beam_kernel_fft = None
        self._beam_fft_shape = None

        self._add_zero = False
        self._vrot_r0_removed_idx = None
        self._full_freepar_idx = None

        if beam is not None:
            if isinstance(beam, str):
                self.add_beam_from_fits(beam, normalize=normalize_beam)
            elif isinstance(beam, bool) and beam:
                normalize = "sum" if normalize_beam is None else normalize_beam
                self.build_beam_from_header(normalize=normalize)
            else:
                raise ValueError("Invalid value for 'beam' argument.")


    def initialize(self, rsep, rnum, init={}, set_options={}, add_zero=False):
        self._add_zero = add_zero

        self.radii = np.linspace(
            0.50 * rsep, 0.50 * rsep * (1.00 + 2.00 * (rnum - 1.00)), rnum
        )
        self.rsep = rsep

        if add_zero:
            self.radii = np.concatenate([[0.0], self.radii])

        self.init(radii=self.radii, **init)

        self.set_options(**set_options)


    def _setup(self, freepar, **kwargs):
        super()._setup(freepar, **kwargs)

        if self._add_zero and "vrot" in self.freepar_idx and len(self.freepar_idx["vrot"]) > 1:
            self._full_freepar_idx = {k: v.copy() for k, v in self.freepar_idx.items()}

            removed_idx = int(self.freepar_idx["vrot"][0])
            self._vrot_r0_removed_idx = removed_idx

            # Drop the r=0 entry from vrot's index list
            self.freepar_idx["vrot"] = self.freepar_idx["vrot"][1:]

            # Shift down all theta indices that sit above the removed slot
            for key in self.freepar_idx:
                self.freepar_idx[key] = np.where(
                    self.freepar_idx[key] > removed_idx,
                    self.freepar_idx[key] - 1,
                    self.freepar_idx[key],
                )

            self.freepar_names = [n for n in self.freepar_names if n != "vrot1"]
            self.ndim -= 1


    def _update_rings(self, rings, theta):
        if self._vrot_r0_removed_idx is not None:
            # Reinsert vrot=0 at the r=0 position before delegating to parent
            theta = np.insert(theta, self._vrot_r0_removed_idx, 0.0)

            saved_idx = self.freepar_idx
            self.freepar_idx = self._full_freepar_idx
            result = super()._update_rings(rings, theta)
            self.freepar_idx = saved_idx
            return result

        return super()._update_rings(rings, theta)


    # -------------------------------------------------------------------------
    # - Import beam kernel from a FITS
    # -------------------------------------------------------------------------
    def add_beam_from_fits(self, fname, normalize=None):
        with fits.open(fname) as hdu:
            kernel = hdu[0].data.copy()

        # Strip degenerate leading axes (e.g. FITS shape (1, ky, kx) → (ky, kx))
        while kernel.ndim > 2 and kernel.shape[0] == 1:
            kernel = kernel[0]

        self._beam_kernel = kernel

        if normalize is not None:
            self._normalize_kernel(normalize)


    # -------------------------------------------------------------------------
    # - Build beam kernel from FITS header (using BBarolo's convention)
    # -------------------------------------------------------------------------
    def build_beam_from_header(self, normalize=None):
        self._beam_kernel = self._build_bb_beam()

        if normalize is not None:
            self._normalize_kernel(normalize)


    def _build_single_kernel(self, bmaj, bmin, bpa, pixsizeX, pixsizeY):
        """Build one 2D Gaussian kernel matching BBarolo's cfield construction."""
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

        ii, jj = np.meshgrid(np.arange(-Xmax, Xmax + 1), np.arange(-Ymax, Ymax + 1))
        x, y = ii * pixsizeX, jj * pixsizeY

        xr = x * cs + y * sn
        yr = -x * sn + y * cs

        argfac = -4.0 * np.log(2.0)
        argX = (xr / bmaj) if bmaj != 0 else np.zeros_like(xr)
        argY = (yr / bmin) if bmin != 0 else np.zeros_like(yr)
        kernel = np.exp(argfac * (argX**2 + argY**2))
        kernel[kernel < 1e-4] = 0
        return kernel


    def _build_bb_beam(self):
        """Build beam kernel(s) matching BBarolo's cfield construction.

        Returns a 2D array (ky, kx) for a uniform beam, or a 3D array
        (nchans, ky, kx) when the beam varies across the frequency axis
        (read from a BEAMS binary-table extension, CASA convention).
        """
        hdr = fits.getheader(self.inp.fname)
        pixsizeX = abs(hdr["CDELT1"]) * 3600  # degrees -> arcsec
        pixsizeY = abs(hdr["CDELT2"]) * 3600  # degrees -> arcsec

        # Per-channel beam: CASA writes a BEAMS binary-table extension
        try:
            with fits.open(self.inp.fname) as hdul:
                beams = hdul["BEAMS"].data
            bmaj_arr = beams["BMAJ"] * 3600  # degrees -> arcsec
            bmin_arr = beams["BMIN"] * 3600
            bpa_arr  = beams["BPA"]
        except KeyError:
            bmaj_arr = np.array([hdr["BMAJ"] * 3600])
            bmin_arr = np.array([hdr["BMIN"] * 3600])
            bpa_arr  = np.array([hdr.get("BPA", 0)])

        kernels = [
            self._build_single_kernel(bmaj, bmin, bpa, pixsizeX, pixsizeY)
            for bmaj, bmin, bpa in zip(bmaj_arr, bmin_arr, bpa_arr)
        ]

        if len(kernels) == 1:
            return kernels[0]

        max_ky = max(k.shape[0] for k in kernels)
        max_kx = max(k.shape[1] for k in kernels)

        padded = []
        for k in kernels:
            pad_y = (max_ky - k.shape[0]) // 2
            pad_x = (max_kx - k.shape[1]) // 2
            padded.append(np.pad(k, ((pad_y, max_ky - k.shape[0] - pad_y),
                                     (pad_x, max_kx - k.shape[1] - pad_x))))

        return np.stack(padded)


    def _normalize_kernel(self, normalize="sum"):
        if isinstance(normalize, bool):
            normalize = "sum" if normalize else None

        if (
            normalize is not None 
            and isinstance(normalize, str)
        ):
            if normalize == "sum":
                if self._beam_kernel.ndim == 2:
                    self._beam_kernel /= self._beam_kernel.sum()
                else:
                    self._beam_kernel /= self._beam_kernel.sum(axis=(-2, -1), keepdims=True)
            elif normalize == "peak":
                if self._beam_kernel.ndim == 2:
                    self._beam_kernel /= self._beam_kernel.max()
                else:
                    self._beam_kernel /= self._beam_kernel.max(axis=(-2, -1), keepdims=True)


    def _build_fft_beam(self):
        _, ny, nx = self.data.shape
        ky, kx = self._beam_kernel.shape[-2], self._beam_kernel.shape[-1]

        fshape_y = ny + ky - 1
        fshape_x = nx + kx - 1

        self._beam_fft_shape = (fshape_y, fshape_x)
        self._beam_kernel_fft = np.fft.rfft2(
            self._beam_kernel, s=self._beam_fft_shape, axes=(-2, -1)
        )


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
    def __init__(self,
        bbobj,
        free_params,
        likelihood=Normal,
        method_norm="constant",
        output=None
    ):
        self.bbobj = bbobj
        self.free_params = free_params
        self.method_norm = method_norm
        self.output = output

        if self.output is None:
            self.output = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.bbobj._opts.add_params(sm=False)

        self.bbobj._setup(self.free_params, useBBres=False)

        self.bbobj.update_prof = any(
            sub in string
            for string in self.bbobj.freepar_names
            for sub in ["inc", "phi", "xpos", "ypos"]
        )

        if self.bbobj.mask is None:
            self.bbobj.mask = np.ones_like(self.bbobj.data, dtype=bool)

        if not self.bbobj.useNorm:
            raise ValueError(
                "modBBarolo does not support setting 'dens' as a free parameter."
            )

        self.freepar_names = list(self.bbobj.freepar_names)
        self.freepar_idx = dict(self.bbobj.freepar_idx)
        self.prior_distr = dict(self.bbobj.prior_distr)

        if self.method_norm == "constant":
            self.freepar_names.append("norm")
            self.freepar_idx["norm"] = len(self.freepar_names) - 1

            distr_kwargs = {
                key: value
                for key, value in self.bbobj.priors["norm"].items()
                if key != "name"
            }

            self.prior_distr["norm"] = get_distribution(
                self.bbobj.priors["norm"]["name"], **distr_kwargs
            )

        elif self.method_norm == "exponential":
            hdr = fits.getheader(self.bbobj.inp.fname)
            self.pixscale = abs(hdr["CDELT2"]) * 3600
            del hdr

            for pname in ("norm", "rdisk"):
                self.freepar_names.append(pname)
                self.freepar_idx[pname] = len(self.freepar_names) - 1

                distr_kwargs = {
                    key: value
                    for key, value in self.bbobj.priors[pname].items()
                    if key != "name"
                }
                self.prior_distr[pname] = get_distribution(
                    self.bbobj.priors[pname]["name"], **distr_kwargs
                )

        if self.bbobj._beam_kernel is None:
            print(
                "Warning. No beam kernel has been built. If your data is beam-convolved, "
                "please run 'bbobj.build_beam_from_header()' before loading the sampler."
            )
        else:
            self.bbobj._build_fft_beam()

        self._likelihood = likelihood if not isinstance(likelihood, type) else likelihood()

    # -----------------------------------------------------------------------------
    # - Model smoothing
    # -----------------------------------------------------------------------------
    def _smooth_model(self, model):
        """Convolve each channel with the beam kernel (pure Python)."""
        model[np.isnan(model)] = 0.00
        
        if self.bbobj._beam_kernel_fft is not None:
            _, ny, nx = model.shape
            ky, kx = self.bbobj._beam_kernel.shape[-2], self.bbobj._beam_kernel.shape[-1]
            model_fft = np.fft.rfft2(model, s=self.bbobj._beam_fft_shape, axes=(-2, -1))
            conv = np.fft.irfft2(
                model_fft * self.bbobj._beam_kernel_fft,
                s=self.bbobj._beam_fft_shape,
                axes=(-2, -1),
            )
            y0 = (ky - 1) // 2
            x0 = (kx - 1) // 2
            model = conv[:, y0 : y0 + ny, x0 : x0 + nx]
        return model

    # -----------------------------------------------------------------------------
    # - Model normalization
    # -----------------------------------------------------------------------------
    def _normalize_model(self, model, data, **kwargs):
        if self.method_norm == "model":
            return model * np.nansum(data) / np.nansum(model)
        elif self.method_norm == "constant":
            return kwargs["norm"] * model
        elif self.method_norm == "exponential":
            norm = kwargs["norm"]
            rdisk = kwargs["rdisk"]
            rings = kwargs["rings"]
            bhi, blo = kwargs["bhi"], kwargs["blo"]

            x0 = np.mean(rings.r["xpos"])
            y0 = np.mean(rings.r["ypos"])
            inc = np.radians(np.mean(rings.r["inc"]))
            phi = np.radians(np.mean(rings.r["phi"]))

            x = np.arange(blo[0], bhi[0])
            y = np.arange(blo[1], bhi[1])
            xx, yy = np.meshgrid(x, y)

            xr = -(xx - x0) * np.sin(phi) + (yy - y0) * np.cos(phi)
            yr = (-(xx - x0) * np.cos(phi) - (yy - y0) * np.sin(phi)) / np.cos(inc)
            R = np.sqrt(xr**2 + yr**2) * self.pixscale

            return model * norm * np.exp(-R / rdisk)

    # -----------------------------------------------------------------------------
    # - Prior Transform for nested sampler
    # -----------------------------------------------------------------------------
    def _prior_transform(self, u):
        p = np.zeros_like(u)
        for key in self.freepar_idx.keys():
            p[self.freepar_idx[key]] = self.prior_distr[key].ppf(
                u[self.freepar_idx[key]]
            )
        return p

    # -----------------------------------------------------------------------------
    # - Build BBarolo model and data+mask for likelihood
    # ------------------------------------------------------------------------------
    def _get_model(self, theta, convolve=False):
        for k in self.freepar_idx:
            if k.startswith(is_positive) and np.any(theta[self.freepar_idx[k]] < 0):
                return -np.inf

        rings = self.bbobj._update_rings(self.bbobj._inri, theta)

        if self.bbobj.update_prof:
            self.bbobj._update_profile(rings)

        # Calculate the model and the boundaries
        model_, bhi, blo, galmod = self.bbobj._calculate_model(rings)

        # Calculate the residuals
        mask = self.bbobj.mask.copy()
        data = self.bbobj.data.copy()

        data_ = data[:, blo[1] : bhi[1], blo[0] : bhi[0]].copy()

        kwargs = {}
        if self.method_norm == "constant":
            kwargs["norm"] = theta[self.freepar_idx["norm"]]
        elif self.method_norm == "exponential":
            kwargs["norm"] = theta[self.freepar_idx["norm"]]
            kwargs["rdisk"] = theta[self.freepar_idx["rdisk"]]
            kwargs["rings"] = rings
            kwargs["bhi"], kwargs["blo"] = bhi, blo

        model_ = self._normalize_model(model_, data_, **kwargs)

        model = np.zeros(data.shape)
        model[:, blo[1] : bhi[1], blo[0] : bhi[0]] = model_.copy()

        if convolve:
            model = self._smooth_model(model)

        libBB.Galmod_delete(galmod)

        return model, data, mask

    # -----------------------------------------------------------------------------
    # - Main method to run the sampler
    # -----------------------------------------------------------------------------
    def run(
        self, method="nautilus", checkpoint=False, resume=False, threads=1, likelihood_kwargs={}, **kwargs
    ):
        for key, value in self._likelihood._build(self.bbobj.data, **likelihood_kwargs).items():
            setattr(self, key, value)
        self._log_likelihood = types.MethodType(self._likelihood._compute.__func__, self)

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

            self.sampler.run(
                verbose=True, discard_exploration=discard_exploration, **run_kwargs
            )

            self.samples, weights, _ = self.sampler.posterior()
            self.weights = np.exp(weights - np.max(weights))

        if pool is not None:
            pool.close()
            pool.join()

        self.params = np.array(
            [corner.quantile(s, 0.50, weights=self.weights)[0] for s in self.samples.T]
        )

        self.bbobj.params = self.params[: len(self.bbobj.freepar_names)]
        self.bbobj.modCalculated = True

    # -----------------------------------------------------------------------------
    # - Save corner plot
    # -----------------------------------------------------------------------------
    def save_corner(self, sigma=10.00, **kwargs):
        if sigma is not None:
            edges = np.array(
                [
                    corner.quantile(s, [0.16, 0.50, 0.84], weights=self.weights)
                    for s in self.samples.T
                ]
            )
            edges = np.array(
                [
                    [
                        np.maximum(
                            e[1] - sigma * (e[1] - e[0]), self.samples[:, i].min()
                        ),
                        np.minimum(
                            e[1] + sigma * (e[2] - e[1]), self.samples[:, i].max()
                        ),
                    ]
                    for i, e in enumerate(edges)
                ]
            )
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
            **corner_kwargs,
        )

        plt.savefig(f"{self.output}_corner.pdf", format="pdf", dpi=300)
        plt.close()

    # -----------------------------------------------------------------------------
    # - Save best-fit model and outputs using the best-fit parameters
    # -----------------------------------------------------------------------------
    def save_best_model(self, plots=True, **kwargs):
        if not self.bbobj.modCalculated:
            print(
                "Sampler has not been run yet. Please run compute() before running this function."
            )
            return

        model, _, _ = self._get_model(self.params, convolve=True)

        rings = self.bbobj._update_rings(self.bbobj._inri, self.params)

        if self.bbobj.update_prof:
            self.bbobj._update_profile(rings)

        libBB.Galfit_setOutRings(self.bbobj._galfit, rings._rings)

        _, ys, xs = self.bbobj.data.shape
        bhi = (ctypes.c_int * 2)(xs, ys)
        blo = (ctypes.c_int * 2)(0, 0)

        galmod = libBB.Galfit_getModel(self.bbobj._galfit, rings._rings, bhi, blo, True)

        buffer = reshapePointer(libBB.Galmod_array(galmod), model.shape)
        buffer[:] = model

        libBB.Galfit_writeOutputs(
            self.bbobj._galfit, galmod, self.bbobj._ellprof, plots
        )
        libBB.Galmod_delete(galmod)
        plt.close()
