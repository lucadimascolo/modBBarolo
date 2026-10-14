import matplotlib.pyplot as plt

from pyBBarolo.bayesian import get_distribution
from pyBBarolo.bayesian import libBB, ctypes, reshapePointer
from pyBBarolo.bayesian import BayesianBBarolo, Rings

import nautilus
import emcee
import corner

import inspect
import types
from datetime import datetime

import numpy as np
import scipy.fft
from astropy.io import fits
from astropy import units

from .likelihood import Normal, NormalRI

import multiprocess

_mp_context = multiprocess.get_context("fork")

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
    "rsep",
)

_active_sampler = None

_norms = ["flux", "local", "constant", "exponential"]

_methods = ["nautilus", "emcee"]


# -----------------------------------------------------------------------------
# Initialize BBarolo
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Signed pixel size (arcsec) along x and y, read as BBarolo does
# -----------------------------------------------------------------------------
def _pixel_size(fitsname):
    hdr = fits.getheader(fitsname)
    if "CDELT1" in hdr and "CDELT2" in hdr:
        return 3600.00 * np.array([hdr["CDELT1"], hdr["CDELT2"]], dtype=np.float64)
    if "CD1_1" in hdr and "CD2_2" in hdr:
        # BBarolo falls back to the CD matrix only when no axis has a CDELT
        if any(key.startswith("CDELT") for key in hdr):
            raise ValueError(
                "The header mixes CDELT keywords with a CD matrix, which BBarolo cannot read: "
                "give CDELT1 and CDELT2, or remove every CDELT keyword."
            )
        return 3600.00 * np.array([hdr["CD1_1"], hdr["CD2_2"]], dtype=np.float64)
    raise ValueError("The header has neither CDELT1/CDELT2 nor CD1_1/CD2_2: the pixel size is unknown.")


class Init(BayesianBBarolo):
    def __init__(self, fitsname, beam=None, normalize_beam=None, *args, **kwargs):
        self._pixsize = _pixel_size(fitsname)
        if self._pixsize[0] > 0.00:
            print("Warning. CDELT1 is positive: BBarolo expects RA to decrease along x and does not handle it consistently.")

        super().__init__(fitsname, *args, **kwargs)

        self._beam_kernel = None
        self._beam_kernel_fft = None
        self._beam_fft_shape = None
        self._beam_crop_start = None

        self._add_zero = False
        self._cdens_boost = None
        self._pad_rings = None

        # Galmod needs ring nodes more than 0.75 pixels apart
        self._min_node_sep = 0.75 * np.abs(self._pixsize).min()
        self._vrot_r0_removed_idx = None
        self._full_freepar_idx = None

        if beam is not None and beam is not False:
            if isinstance(beam, str):
                self.add_beam_from_fits(beam, normalize=normalize_beam)
            elif isinstance(beam, bool) and beam:
                normalize = "sum" if normalize_beam is None else normalize_beam
                self.build_beam_from_header(normalize=normalize)
            else:
                raise ValueError("Invalid value for 'beam' argument.")


    def initialize(self, rsep, rnum, init={}, set_options={}, add_zero=False,
                   cdens_boost=None):
    
        self._add_zero = add_zero

        self.radii = np.linspace(
            0.50 * rsep, 0.50 * rsep * (1.00 + 2.00 * (rnum - 1.00)), rnum
        )
        self.rsep = rsep

        if add_zero:
            self.radii = np.concatenate([[0.0], self.radii])

        if len(self.radii) < 2:
            raise ValueError("At least two rings are required: use rnum >= 2 or add_zero=True.")

        if 0.50 * rsep <= self._min_node_sep:
            raise ValueError(
                f"'rsep' must be larger than {2.00 * self._min_node_sep:.4g} arcsec (1.5 pixels)."
            )

        init = dict(init)

        if cdens_boost is not None:
            if isinstance(cdens_boost, dict):
                self._cdens_boost = np.ones(len(self.radii))
                for cdens_boost_idx in cdens_boost.keys():
                    if cdens_boost[cdens_boost_idx] is not None:
                        self._cdens_boost[cdens_boost_idx] *= cdens_boost[cdens_boost_idx]
            else:
                raise ValueError("'cdens_boost'should be a dict.")

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
            rings = super()._update_rings(rings, theta)
            self.freepar_idx = saved_idx
        else:
            rings = super()._update_rings(rings, theta)

        # vrot(r=0) is zero also when vrot is fixed, single or functional
        if self._add_zero and rings.r["vrot"][0] != 0.00:
            vrot = np.array(rings.r["vrot"], copy=True)
            vrot[0] = 0.00
            rings.modify_parameter("vrot", vrot, makeobj=True)

        return rings


    # -------------------------------------------------------------------------
    # - Ring densities from Ellprof, without sign flips or an empty centre
    # -------------------------------------------------------------------------
    def _update_profile(self, rings):
        libBB.Ellprof_update_rings(self._ellprof, rings._rings)
        libBB.Ellprof_compute(self._ellprof)
        dens = reshapePointer(libBB.Ellprof_dens_array(self._ellprof), (1, rings.nr))[0]
        dens = np.array(dens, dtype=np.float64)

        # The annulus of a ring at r=0 can hold no pixel
        if rings.nr > 1 and rings.r["radii"][0] == 0.00 and not np.abs(dens[0]) > 0.00:
            dens[0] = dens[1]

        # Negative ring means are floored: Galmod skips rings with zero density
        peak = np.nanmax(dens) if np.any(np.isfinite(dens)) else 0.00
        floor = 1.00e-03 * max(peak, 0.00)
        dens = np.where(np.isfinite(dens) & (dens > floor), dens, floor)

        rings.modify_parameter("dens", dens, makeobj=True)


    # -------------------------------------------------------------------------
    # - Rings handed to Galmod, with explicit nodes at the model edges
    # -------------------------------------------------------------------------
    def _padded_rings(self, rings):
        radii = np.asarray(rings.r["radii"], dtype=np.float64)

        # Ring centres sit at odd multiples of rsep/2
        inner = radii[0] > 0.00
        rsep = 2.00 * (radii[0] if inner else radii[1])

        def pad(values):
            values = np.asarray(values)
            return np.concatenate([values[:1] if inner else values[:0], values, values[-1:]])

        padded = {key: pad(values) for key, values in rings.r.items()}
        padded["radii"] = np.concatenate(
            [[0.00] if inner else [], radii, [radii[-1] + 0.50 * rsep]]
        )

        if self._pad_rings is None or self._pad_rings.nr != len(padded["radii"]):
            self._pad_rings = Rings(len(padded["radii"]))
        for key, values in padded.items():
            self._pad_rings.modify_parameter(key, values)
        self._pad_rings.make_object()

        return self._pad_rings


    # -------------------------------------------------------------------------
    # - Render the rings with their values at the nominal radii
    # -------------------------------------------------------------------------
    def _render_model(self, rings, fullcube=False):
        # finalModel=False: BBarolo would otherwise shift the first and last ring
        padded = self._padded_rings(rings)

        _, ys, xs = self.data.shape
        if fullcube:
            bhi, blo = (ctypes.c_int * 2)(xs, ys), (ctypes.c_int * 2)(0, 0)
        else:
            bhi, blo = (ctypes.c_int * 2)(0, 0), (ctypes.c_int * 2)(0, 0)
            libBB.Galfit_getModelSize(self._galfit, padded._rings, bhi, blo)

        galmod = libBB.Galfit_getModel(self._galfit, padded._rings, bhi, blo, False)
        bhi, blo = np.array(bhi), np.array(blo)

        shape = (self.inp.dim[2], bhi[1] - blo[1], bhi[0] - blo[0])
        mod = reshapePointer(libBB.Galmod_array(galmod), shape)

        return mod, bhi, blo, galmod


    # -------------------------------------------------------------------------
    # - Calculate model, drawing more clouds in the cdens_boost rings
    # -------------------------------------------------------------------------
    def _calculate_model(self, rings, fullcube=False):
        boost = self._cdens_boost
        if boost is None or np.all(boost == 1.00):
            return self._render_model(rings, fullcube=fullcube)

        options = {str(k).lower(): v for k, v in self._opts.opts.items()}
        if int(options.get("cmode", 1)) != 1:
            raise ValueError("'cdens_boost' requires cmode=1.")
        cdens = float(options.get("cdens", 10.00))

        # Boosted rings rendered apart with dens*scale, then divided by scale
        dens_saved = np.array(rings.r["dens"], dtype=np.float32, copy=True)
        boosted = 0.00

        # Galmod never draws fewer than 10 clouds per pixel: boost from that floor
        clouds = cdens * dens_saved.astype(np.float64)
        floor = np.divide(10.00, clouds, out=np.ones_like(clouds), where=clouds > 0.00)
        scale = np.where(boost != 1.00, boost * np.maximum(floor, 1.00), 1.00)

        try:
            for factor in np.unique(scale[boost != 1.00]):
                dens = np.where((boost != 1.00) & (scale == factor), dens_saved * factor, 0.00)
                rings.modify_parameter("dens", dens, makeobj=True)

                mod, _, _, galmod = self._render_model(rings, fullcube=fullcube)
                boosted = boosted + mod / factor
                libBB.Galmod_delete(galmod)

            dens = np.where(boost == 1.00, dens_saved, 0.00)
            rings.modify_parameter("dens", dens, makeobj=True)

            mod, bhi, blo, galmod = self._render_model(rings, fullcube=fullcube)
            mod += boosted
            return mod, bhi, blo, galmod
        finally:
            rings.modify_parameter("dens", dens_saved, makeobj=True)


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
        pixsizeX, pixsizeY = np.abs(self._pixsize)

        # Per-channel beam: CASA writes a BEAMS binary-table extension
        try:
            with fits.open(self.inp.fname) as hdul:
                beams = hdul["BEAMS"]

                def _column(name, unit, default):
                    colunit = beams.columns[name].unit or default
                    values = np.array(beams.data[name], dtype=float)
                    return (values * units.Unit(colunit)).to_value(unit)

                bmaj_arr = _column("BMAJ", "arcsec", "arcsec")
                bmin_arr = _column("BMIN", "arcsec", "arcsec")
                bpa_arr  = _column("BPA", "deg", "deg")

                # Rows are not guaranteed to be in channel order
                if "CHAN" in beams.columns.names:
                    order = np.argsort(np.asarray(beams.data["CHAN"]), kind="stable")
                    bmaj_arr, bmin_arr, bpa_arr = bmaj_arr[order], bmin_arr[order], bpa_arr[order]
        except KeyError:
            bmaj_arr = np.array([hdr["BMAJ"] * 3600])
            bmin_arr = np.array([hdr["BMIN"] * 3600])
            bpa_arr  = np.array([hdr.get("BPA", 0)])

        if len(bmaj_arr) not in (1, self.inp.dim[2]):
            raise ValueError(
                f"The BEAMS table has {len(bmaj_arr)} rows for {self.inp.dim[2]} channels; "
                "only one beam per channel (single polarization) is supported."
            )

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
            else:
                raise ValueError(f"Invalid value for 'normalize': {normalize!r}. Please use 'sum', 'peak' or None.")


    # -------------------------------------------------------------------------
    # - Crop the kernel to the largest window symmetric about its peak
    # -------------------------------------------------------------------------
    def _symmetrize_beam(self):
        stack = self._beam_kernel.ndim == 3
        kernel = self._beam_kernel if stack else self._beam_kernel[None]
        ny, nx = kernel.shape[-2:]

        peaks = [np.unravel_index(np.argmax(k), k.shape) for k in kernel]
        hy = min(min(y0, ny - 1 - y0) for y0, _ in peaks)
        hx = min(min(x0, nx - 1 - x0) for _, x0 in peaks)
        if hy < 1 or hx < 1:
            raise ValueError("The beam kernel peaks on the edge of its stamp: it cannot be symmetrized.")

        kernel = np.stack([
            k[y0 - hy : y0 + hy + 1, x0 - hx : x0 + hx + 1] for k, (y0, x0) in zip(kernel, peaks)
        ])

        if kernel.shape[-2:] != (ny, nx):
            print(f"Beam kernel cropped from {(ny, nx)} to {kernel.shape[-2:]} to be symmetric about its peak.")

        asym = np.abs(kernel - kernel[:, ::-1, ::-1]).max() / np.abs(kernel).max()
        if asym > 1.00e-03:
            print(f"Warning. The beam kernel is not symmetric about its peak (up to {asym:.1e} of the peak).")

        self._beam_kernel = kernel if stack else kernel[0]


    def _build_fft_beam(self, center="stamp"):
        nchans, ny, nx = self.data.shape
        if self._beam_kernel.ndim == 3 and self._beam_kernel.shape[0] != nchans:
            raise ValueError(
                f"The beam kernel has {self._beam_kernel.shape[0]} channels, but the data have {nchans}."
            )
        image_shape = (ny, nx)
        kernel_shape = self._beam_kernel.shape[-2:]

        # Fast FFT length
        self._beam_fft_shape = tuple(
            scipy.fft.next_fast_len(int(i + k - 1), real=True)
            for i, k in zip(image_shape, kernel_shape)
        )

        self._beam_kernel_fft = np.fft.rfft2(
            self._beam_kernel, s=self._beam_fft_shape, axes=(-2, -1)
        )

        # Crop-back offset
        if center == "stamp":
            # Stamp centre, with a half-pixel shift for even sizes
            stamp = [0.50 * (k - 1) for k in kernel_shape]
            start = [int(np.ceil(c)) for c in stamp]
            shift = [s - c for s, c in zip(start, stamp)]

            if any(shift):
                fy = np.fft.fftfreq(self._beam_fft_shape[0])[:, None]
                fx = np.fft.rfftfreq(self._beam_fft_shape[1])[None, :]
                ramp = np.exp(-2.00j * np.pi * (fy * shift[0] + fx * shift[1]))
                self._beam_kernel_fft = self._beam_kernel_fft * ramp

            self._beam_crop_start = tuple(start)
        elif center == "peak":
            # Kernel peak, per channel
            if self._beam_kernel.ndim == 2:
                self._beam_crop_start = np.unravel_index(
                    np.argmax(self._beam_kernel), kernel_shape
                )
            else:
                flat_idx = self._beam_kernel.reshape(self._beam_kernel.shape[0], -1).argmax(axis=1)
                self._beam_crop_start = np.unravel_index(flat_idx, kernel_shape)
        else:
            raise ValueError("Invalid value for 'center' argument.")


# -----------------------------------------------------------------------------
# Multiprocessing handlers
# -----------------------------------------------------------------------------
def _mp_log_likelihood(theta):
    return _active_sampler._log_likelihood(theta)

def _mp_prior_transform(u):
    return _active_sampler._prior_transform(u)

def _mp_log_posterior(theta):
    return _active_sampler._log_posterior(theta)

# -----------------------------------------------------------------------------
# Sampler class
# -----------------------------------------------------------------------------
class Sampler:
    def __init__(self,
        bbobj,
        free_params,
        likelihood=Normal,
        method_norm="constant",
        output=None,
        moment_zero=None,
        regularize={"vdisp": 0.00}
    ):
        if bbobj._galfit is not None:
            raise RuntimeError(
                "This Init object already belongs to a Sampler. Please create a new one."
            )

        self.bbobj = bbobj
        self.free_params = free_params
        self._fit_rsep = "rsep" in free_params

        if self._fit_rsep and "radmax" in free_params:
            raise ValueError(
                "'radmax' and 'rsep' cannot both be free: the 'radmax' prior refers to the initial ring grid."
            )

        self.output = output
        if self.output is None:
            self.output = datetime.now().strftime("%Y%m%d_%H%M%S")

        if method_norm not in _norms:
            raise ValueError(f"Method {method_norm} not recognized. Please use one of: {_norms}")
        
        self.method_norm = method_norm

        # If the user set priors["vdisp"]["name"] == "sqrt", vdisp is
        # sampled uniform in vdisp^2 over the given [loc, loc+scale] bounds.
        self._vdisp_sqrt = (
            any(p in ("vdisp", "vdisp_single") for p in free_params)
            and self.bbobj.priors["vdisp"]["name"] == "sqrt"
        )
        if self._vdisp_sqrt:
            distr_kw = {
                key: value
                for key, value in self.bbobj.priors["vdisp"].items()
                if key != "name"
            }
            self.bbobj.prior_distr["vdisp"] = get_distribution("uniform", **distr_kw)

        # If the user set priors["vdisp"]["name"] == "constrained", vdisp is
        # reparameterized as vdisp_i = vdisp_mu + tau * cumsum(z)_i, z_i ~ N(0, 1).
        self._vdisp_constrained = (
            any(p.split("_")[0] == "vdisp" for p in free_params)
            and self.bbobj.priors["vdisp"]["name"] == "constrained"
        )
        if self._vdisp_constrained:
            if "vdisp" not in free_params:
                raise ValueError(
                    "The 'constrained' vdisp prior mode requires the bare 'vdisp' "
                    "free parameter (one value per ring), not 'vdisp_single'/'vdisp_func'."
                )
            if "radmax" in free_params:
                raise ValueError(
                    "'radmax' cannot be free with the 'constrained' vdisp prior mode: "
                    "the random-walk steps are scaled to the initial ring grid."
                )
            self.bbobj.prior_distr["vdisp"] = get_distribution("norm", loc=0.00, scale=1.00)

        self.bbobj._opts.add_params(sm=False)

        self.bbobj._setup(
            [p for p in self.free_params if p != "rsep"], useBBres=False
        )

        self.bbobj.update_prof = self._fit_rsep or any(
            sub in string
            for string in self.bbobj.freepar_names
            for sub in ["inc", "phi", "xpos", "ypos", "radmax"]
        )

        if self.bbobj.mask is None:
            self.bbobj.mask = np.ones_like(self.bbobj.data, dtype=bool)

        # BBarolo reads blanked voxels as zeros: restore them for the likelihood
        self.data = self.bbobj.data
        with fits.open(self.bbobj.inp.fname) as hdu:
            raw = np.squeeze(hdu[0].data)
        if raw.shape != self.data.shape:
            print("Warning. Cannot match the FITS data to BBarolo's cube: blanked voxels are kept as zeros.")
        elif not np.all(np.isfinite(raw)):
            self.data = np.where(np.isfinite(raw), self.data, np.nan)
        del raw

        self.moment_zero = moment_zero
        self.moment_zero_data = np.nansum(self.bbobj.data * self.bbobj.mask, axis=0)

        if not self.bbobj.useNorm:
            raise ValueError(
                "modBBarolo does not support setting 'dens' as a free parameter."
            )

        self.freepar_names = list(self.bbobj.freepar_names)
        self.freepar_idx = dict(self.bbobj.freepar_idx)
        self._func_params = sorted({k.split("_p")[0] for k in self.freepar_idx if "_p" in k})
        self.prior_distr = dict(self.bbobj.prior_distr)

        if self._fit_rsep:
            self.freepar_names.append("rsep")
            self.freepar_idx["rsep"] = len(self.freepar_names) - 1

            distr_kwargs = {
                key: value
                for key, value in self.bbobj.priors["rsep"].items()
                if key != "name"
            }
            self.prior_distr["rsep"] = get_distribution(
                self.bbobj.priors["rsep"]["name"], **distr_kwargs
            )

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
            # The deprojection below assumes Galmod's standard grid
            pixsize = self.bbobj._pixsize
            if pixsize[0] > 0.00 or not np.isclose(abs(pixsize[0]), abs(pixsize[1]), rtol=1.00e-06):
                raise ValueError(
                    "The 'exponential' normalization requires square pixels and a negative CDELT1."
                )
            self.pixscale = abs(pixsize[1])

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

        if self._vdisp_constrained:
            for pname in ("vdisp_mu", "tau"):
                if pname not in self.bbobj.priors:
                    raise ValueError(
                        f"The 'constrained' vdisp prior mode requires priors['{pname}'] "
                        "to be set before constructing Sampler."
                    )
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

        # Physical layout, as seen by _update_rings and the rest of the model
        self._phys_idx = self.freepar_idx
        self._phys_names = list(self.freepar_names)

        if self._vdisp_constrained:
            # vdisp_1 = vdisp_mu: the first ring has no innovation of its own
            first = int(self.freepar_idx["vdisp"][0])
            self._phys_idx = dict(self.freepar_idx)

            self.freepar_idx = {}
            for key, value in self._phys_idx.items():
                value = np.asarray(value)
                if key == "vdisp":
                    value = value[1:]
                value = np.where(value > first, value - 1, value)
                self.freepar_idx[key] = value if value.ndim else int(value)

            innovations = set(int(i) for i in self._phys_idx["vdisp"][1:])
            self.freepar_names = [
                f"z_{name}" if i in innovations else name
                for i, name in enumerate(self._phys_names) if i != first
            ]

            # Random-walk variance proportional to the radial step
            steps = np.diff(np.asarray(self.bbobj.radii, dtype=np.float64))
            self._walk_weight = np.sqrt(steps / steps.max())

        self._likelihood = likelihood if not isinstance(likelihood, type) else likelihood()

        # Reference maps: the dirty data cannot normalize the clean model of NormalRI
        if self.method_norm in ["flux", "local"] and self.moment_zero is None:
            if isinstance(self._likelihood, NormalRI):
                raise ValueError(
                    f"'NormalRI' with '{self.method_norm}' requires a clean 'moment_zero': the dirty "
                    "data overestimate the flux by the PSF sum. Pass one, or use 'constant'."
                )
            if self.method_norm == "local":
                print(
                    "Warning. 'local' without a 'moment_zero' scales the unconvolved model "
                    "against the beam-convolved data."
                )

        if self.method_norm == "flux":
            reference = self.moment_zero_data if self.moment_zero is None else self.moment_zero
            self._reference_flux = float(np.nansum(reference))
            if not self._reference_flux > 0.00:
                raise ValueError(
                    f"The total flux of the reference map is {self._reference_flux:.3g}: 'flux' needs a positive one."
                )

        if self.bbobj._beam_kernel is None:
            print(
                "Warning. No beam kernel has been built. If your data is beam-convolved, "
                "please run 'bbobj.build_beam_from_header()' before loading the sampler."
            )
        else:
            # Same convention as socca
            center = "peak" if isinstance(self._likelihood, NormalRI) else "stamp"
            if center == "peak":
                # A dirty beam is symmetric: its transform must be real
                self.bbobj._symmetrize_beam()

                peak = self.bbobj._beam_kernel.max(axis=(-2, -1))
                if not np.allclose(peak, 1.00, rtol=1.00e-03):
                    print(
                        "Warning. 'NormalRI' expects a dirty beam with a unit peak, "
                        f"but the kernel peaks at {np.max(peak):.3g}."
                    )
            self.bbobj._build_fft_beam(center=center)

        for key in regularize.keys():
            if key not in ["vdisp"]:
                raise ValueError(f"Regularization key '{key}' is not recognized.")
        if self._vdisp_constrained and regularize.get("vdisp", 0.00) != 0.00:
            raise ValueError(
                "regularize['vdisp'] cannot be combined with the 'constrained' "
                "vdisp prior mode; the random walk already imposes ring-to-ring smoothness."
            )
        if regularize.get("vdisp", 0.00) != 0.00 and len(self.freepar_idx.get("vdisp", [])) < 2:
            raise ValueError(
                "regularize['vdisp'] requires the bare 'vdisp' free parameter "
                "(one value per ring) with at least two rings."
            )
        self._regularize = regularize.copy()

    # -----------------------------------------------------------------------------
    # - Model smoothing
    # -----------------------------------------------------------------------------
    def _smooth_model(self, model):
        """Convolve each channel with the beam kernel (pure Python)."""
        model[np.isnan(model)] = 0.00
        
        if self.bbobj._beam_kernel_fft is not None:
            _, ny, nx = model.shape
            model_fft = np.fft.rfft2(model, s=self.bbobj._beam_fft_shape, axes=(-2, -1))
            conv = np.fft.irfft2(
                model_fft * self.bbobj._beam_kernel_fft,
                s=self.bbobj._beam_fft_shape,
                axes=(-2, -1),
            )
            y0, x0 = self.bbobj._beam_crop_start
            if np.ndim(y0) == 0:
                model = conv[:, y0 : y0 + ny, x0 : x0 + nx]
            else:
                model = np.stack([
                    conv[c, y0[c] : y0[c] + ny, x0[c] : x0[c] + nx]
                    for c in range(conv.shape[0])
                ])
        return model

    # -----------------------------------------------------------------------------
    # - Model normalization
    # -----------------------------------------------------------------------------
    def _normalize_model(self, model, **kwargs):
        if self.method_norm in ["flux", "local"]:
            # Unmasked model sums: the mask only enters the default reference
            model_ = np.nansum(model, axis=0)
            if not np.nansum(model_) > 0.00:
                return None

            if self.method_norm == "flux":
                # Total flux of the reference map, independent of the model footprint
                return model * self._reference_flux / np.nansum(model_)

            ratio = np.divide(
                kwargs["moment_zero"], model_, out=np.zeros(model_.shape), where=model_ != 0
            )
            model = model * np.nan_to_num(ratio)
            return model if np.nansum(model) > 0.00 else None

        # No emission to scale: the model is empty
        if not np.nansum(model) > 0.00:
            return None
        if self.method_norm == "constant":
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

            return model * norm * np.exp(-R[None, :, :] / rdisk)

    # -----------------------------------------------------------------------------
    # - Prior Transform for nested sampler
    # -----------------------------------------------------------------------------
    def _prior_transform(self, u):
        p = np.zeros_like(u)
        for key in self.freepar_idx.keys():
            p[self.freepar_idx[key]] = self.prior_distr[key].ppf(
                u[self.freepar_idx[key]]
            )
        if self._vdisp_sqrt:
            # Bounds [a, b] are given directly in vdisp units; sample so that
            # vdisp is uniform in vdisp^2 (density ~ vdisp) over [a, b].
            idx = self.freepar_idx["vdisp"]
            a, b = self.prior_distr["vdisp"].support()
            p[idx] = np.sqrt(a**2 + u[idx] * (b**2 - a**2))
        return p

    # -----------------------------------------------------------------------------
    # - Log-prior for ensemble sampler (emcee)
    # -----------------------------------------------------------------------------
    def _log_prior(self, theta):
        lp = 0.00
        for key, idx in self.freepar_idx.items():
            value = theta[idx]
            if key == "vdisp" and self._vdisp_sqrt:
                # Matches _prior_transform: vdisp uniform in vdisp^2 over [a, b]
                # (bounds given directly in vdisp units) -> pdf(y) = 2y/(b^2-a^2).
                a, b = self.prior_distr["vdisp"].support()
                if np.any(value < a) or np.any(value > b):
                    return -np.inf
                logpdf = np.log(2.00 * value) - np.log(b**2 - a**2)
            else:
                logpdf = self.prior_distr[key].logpdf(value)
            if np.any(~np.isfinite(logpdf)):
                return -np.inf
            lp += np.sum(logpdf)
        return lp

    # -----------------------------------------------------------------------------
    # - Log-probability (prior + likelihood) for ensemble sampler (emcee)
    # -----------------------------------------------------------------------------
    def _log_posterior(self, theta):
        lp = self._log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = self._log_likelihood(theta)
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll

    # -----------------------------------------------------------------------------
    # - Regularization term for velocity dispersion
    # -----------------------------------------------------------------------------
    def _regularize_vdisp(self, theta, alpha=1.00):
        strength = self._regularize.get("vdisp", 0.00)
        if strength == 0.00 or len(self.freepar_idx.get("vdisp", [])) < 2:
            return 0.00
        vdisp = theta[self.freepar_idx["vdisp"]]
        return -strength * np.mean(np.abs(np.diff(vdisp)) ** alpha)

    # -----------------------------------------------------------------------------
    # - Map z-space vdisp (constrained mode) to physical per-ring vdisp
    # -----------------------------------------------------------------------------
    def _physical_theta(self, theta):
        """Map sampled vectors (last axis) to the physical layout. In the 'constrained'
        vdisp mode vdisp_1 = vdisp_mu and vdisp_i = vdisp_mu + tau * sum_{j<=i} w_j z_j,
        with w_j the square root of the radial step in units of the largest one."""
        if not self._vdisp_constrained:
            return theta
        theta = np.asarray(theta, dtype=np.float64)
        phys = np.insert(theta, int(self._phys_idx["vdisp"][0]), 0.00, axis=-1)

        z = theta[..., self.freepar_idx["vdisp"]] * self._walk_weight
        mu = theta[..., self.freepar_idx["vdisp_mu"], None]
        tau = theta[..., self.freepar_idx["tau"], None]
        steps = np.concatenate([np.zeros(z.shape[:-1] + (1,)), z], axis=-1)
        phys[..., self._phys_idx["vdisp"]] = mu + tau * np.cumsum(steps, axis=-1)
        return phys

    # -----------------------------------------------------------------------------
    # - Ring radii for a given separation, as built by Init.initialize()
    # -----------------------------------------------------------------------------
    def _radii(self, rsep):
        nr = len(self.bbobj.radii) - int(self.bbobj._add_zero)
        radii = np.linspace(0.50 * rsep, 0.50 * rsep * (1.00 + 2.00 * (nr - 1.00)), nr)
        return np.concatenate([[0.00], radii]) if self.bbobj._add_zero else radii

    # -----------------------------------------------------------------------------
    # - Build BBarolo model and data+mask for likelihood
    # ------------------------------------------------------------------------------
    def _get_model(self, theta, convolve=False):
        theta = self._physical_theta(theta)

        idx = self._phys_idx
        for k in idx:
            if k in is_positive and np.any(theta[idx[k]] < 0):
                return -np.inf

        if self._fit_rsep:
            rsep_val = theta[idx["rsep"]]
            if 0.50 * rsep_val <= self.bbobj._min_node_sep:
                return -np.inf
            self.bbobj._inri.modify_parameter("radii", self._radii(rsep_val))

        rings = self.bbobj._update_rings(self.bbobj._inri, theta)

        # Functional forms: check the ring values they produce
        for k in self._func_params:
            if k in is_positive and np.any(rings.r[k] < 0):
                return -np.inf

        # BBarolo aborts on rings that are too close or on a centre outside the map
        if np.any(np.diff(rings.r["radii"].astype(np.float64)) <= self.bbobj._min_node_sep * (1.00 + 1.00e-06)):
            return -np.inf
        _, ny, nx = self.data.shape
        xpos, ypos = np.mean(rings.r["xpos"]), np.mean(rings.r["ypos"])
        if not (0.00 <= xpos < nx and 0.00 <= ypos < ny):
            return -np.inf

        if self.bbobj.update_prof:
            self.bbobj._update_profile(rings)

        # Calculate the model and the boundaries
        model_, bhi, blo, galmod = self.bbobj._calculate_model(rings)

        try:
            kwargs = {}
            if self.method_norm == "constant":
                kwargs["norm"] = theta[idx["norm"]]
            elif self.method_norm == "exponential":
                kwargs["norm"] = theta[idx["norm"]]
                kwargs["rdisk"] = theta[idx["rdisk"]]
                kwargs["rings"] = rings
                kwargs["bhi"], kwargs["blo"] = bhi, blo
            else:
                # Reference map: the user-provided one or, by default, the data
                moment_zero = self.moment_zero_data if self.moment_zero is None else self.moment_zero
                kwargs["moment_zero"] = moment_zero[blo[1] : bhi[1], blo[0] : bhi[0]]

            # The normalization always applies to the unconvolved model
            model_ = self._normalize_model(model_, **kwargs)

            # No flux to scale, or a non-finite model
            if model_ is None or not np.all(np.isfinite(model_)):
                return -np.inf

            model = np.zeros(self.data.shape)
            model[:, blo[1] : bhi[1], blo[0] : bhi[0]] = model_
        finally:
            libBB.Galmod_delete(galmod)

        if convolve:
            model = self._smooth_model(model)

        return model, self.data, self.bbobj.mask

    # -----------------------------------------------------------------------------
    # - Build the likelihood and bind it to the sampler
    # -----------------------------------------------------------------------------
    def _setup_likelihood(self, **likelihood_kwargs):
        if "beam" in likelihood_kwargs:
            raise ValueError("'beam' cannot be passed in 'likelihood_kwargs': the kernel comes from the Init object.")
        likelihood_built = self._likelihood._build(
            self.data, beam=self.bbobj._beam_kernel, **likelihood_kwargs
        )
        for key, value in likelihood_built.items():
            setattr(self, key, value)
        self._log_likelihood = types.MethodType(self._likelihood._compute.__func__, self)

    # -----------------------------------------------------------------------------
    # - Checkpoints must match the current parameter layout
    # -----------------------------------------------------------------------------
    def _check_checkpoint(self, filepath, resume, method):
        import h5py
        import os

        # nautilus does not compare the dimensions when resuming
        names = f"{os.path.splitext(filepath)[0]}_params.txt"
        if resume and os.path.exists(filepath):
            stored = None
            if os.path.exists(names):
                with open(names) as f:
                    stored = f.read().split()
            elif method == "nautilus":
                with h5py.File(filepath, "r") as f:
                    if "sampler" in f:
                        stored = int(f["sampler"].attrs["n_dim"])

            current = self.freepar_names if isinstance(stored, list) else len(self.freepar_names)
            if stored is not None and stored != current:
                raise ValueError(
                    f"The checkpoint {filepath} was written for other free parameters ({stored}), "
                    f"not {current}. Delete it, or resume with the parameters it was written for."
                )
        with open(names, "w") as f:
            f.write("\n".join(self.freepar_names) + "\n")

    # -----------------------------------------------------------------------------
    # - Main method to run the sampler
    # -----------------------------------------------------------------------------
    def run(
        self, method="nautilus", checkpoint=False, resume=False, threads=1, likelihood_kwargs={}, **kwargs
    ):

        if method not in _methods:
            raise ValueError(f"Method {method} not recognized. Please use one of: {_methods}")

        self._setup_likelihood(**likelihood_kwargs)

        global _active_sampler
        _active_sampler = self

        ndim = len(self.freepar_names)

        if threads > 1:
            pool = _mp_context.Pool(threads)
        else:
            pool = None

        try:
            if method == "nautilus":
                nlive = 1000
                for key in ["nlive", "n_live"]:
                    nlive = kwargs.pop(key, nlive)

                discard_exploration = kwargs.pop("discard_exploration", True)

                sampler_kwargs = {}
                for key in inspect.signature(nautilus.Sampler).parameters.keys():
                    if key not in [
                        "prior",
                        "likelihood",
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

                if kwargs:
                    raise TypeError(
                        f"Unrecognized arguments for method '{method}': {sorted(kwargs)}"
                    )

                filepath = f"{self.output}_checkpoint.h5" if checkpoint else None
                if checkpoint:
                    self._check_checkpoint(filepath, resume, "nautilus")

                self.sampler = nautilus.Sampler(
                    prior=_mp_prior_transform,
                    likelihood=_mp_log_likelihood,
                    n_live=nlive,
                    n_dim=ndim,
                    filepath=filepath,
                    resume=resume,
                    pool=pool,
                    **sampler_kwargs,
                )

                self.converged = self.sampler.run(
                    verbose=True, discard_exploration=discard_exploration, **run_kwargs
                )
                if not self.converged:
                    print(
                        "Warning. The nautilus run stopped on 'n_like_max' or 'timeout' and has not converged: "
                        "the posterior and the evidence are not reliable."
                    )

                self.samples, weights, self.log_l = self.sampler.posterior()
                self.weights = np.exp(weights - np.max(weights))
                self.log_z = self.sampler.log_z

            elif method == "emcee":
                nwalkers = kwargs.pop("nwalkers", max(4 * ndim, 20))
                nsteps = kwargs.pop("nsteps", 5000)
                discard = kwargs.pop("discard", 0)
                thin = kwargs.pop("thin", 1)
                seed = kwargs.pop("seed", None)

                sampler_kwargs = {}
                for key in inspect.signature(emcee.EnsembleSampler).parameters.keys():
                    if key not in [
                        "nwalkers",
                        "ndim",
                        "log_prob_fn",
                        "pool",
                        "backend",
                        "args",
                        "kwargs",
                    ]:
                        if key in kwargs:
                            sampler_kwargs[key] = kwargs.pop(key)

                run_kwargs = {}
                # run_mcmc only declares **kwargs, forwarded to sample()
                for key in inspect.signature(emcee.EnsembleSampler.sample).parameters.keys():
                    if key not in ["self", "initial_state", "iterations", "progress"]:
                        if key in kwargs:
                            run_kwargs[key] = kwargs.pop(key)

                if kwargs:
                    raise TypeError(
                        f"Unrecognized arguments for method '{method}': {sorted(kwargs)}"
                    )

                if checkpoint:
                    self._check_checkpoint(f"{self.output}_checkpoint_emcee.h5", resume, "emcee")
                    backend = emcee.backends.HDFBackend(f"{self.output}_checkpoint_emcee.h5")
                    if not resume:
                        backend.reset(nwalkers, ndim)
                else:
                    backend = None

                self.sampler = emcee.EnsembleSampler(
                    nwalkers, ndim, _mp_log_posterior, pool=pool, backend=backend, **sampler_kwargs
                )
                self.converged = None

                # Own random streams, leaving the global NumPy state untouched
                rng = np.random if seed is None else np.random.default_rng(seed)

                # 'nsteps' is a total iteration target, not an increment: resuming
                # only runs the remaining iterations needed to reach it.
                completed = backend.iteration if backend is not None and backend.initialized else 0
                remaining = max(nsteps - completed, 0)

                if completed > 0:
                    initial_state = None
                else:
                    if seed is not None:
                        self.sampler._random = np.random.RandomState(seed)
                    initial_state = np.array(
                        [self._prior_transform(rng.uniform(size=ndim)) for _ in range(nwalkers)]
                    )

                if remaining > 0:
                    self.sampler.run_mcmc(initial_state, remaining, progress=True, **run_kwargs)
                else:
                    print(
                        f"Already at {completed} iterations (nsteps={nsteps}); nothing to run."
                    )

                self.samples = self.sampler.get_chain(discard=discard, thin=thin, flat=True)
                self.weights = np.ones(len(self.samples))
                self.log_z = None

                # emcee stores the log-posterior: remove the prior
                log_post = self.sampler.get_log_prob(discard=discard, thin=thin, flat=True)
                with np.errstate(invalid="ignore"):
                    self.log_l = log_post - np.array([self._log_prior(t) for t in self.samples])
        except BaseException:
            if pool is not None:
                pool.terminate()
            raise

        if pool is not None:
            pool.close()
            pool.join()

        self.params = np.array(
            [corner.quantile(s, 0.50, weights=self.weights)[0] for s in self.samples.T]
        )

        # Best samples, next to the vector of marginal medians
        self.log_l = np.where(np.isnan(self.log_l), -np.inf, self.log_l)
        log_prior = np.array([self._log_prior(t) for t in self.samples])
        self.params_maxlike = self.samples[np.argmax(self.log_l)].copy()
        self.params_maxpost = self.samples[np.argmax(self.log_l + log_prior)].copy()

        if self._vdisp_constrained:
            # z values mapping onto the median physical vdisp
            vdisp = self._physical_theta(self.samples)[:, self._phys_idx["vdisp"]]
            vdisp = np.array([corner.quantile(v, 0.50, weights=self.weights)[0] for v in vdisp.T])

            # vdisp_1 = vdisp_mu, so its median is already the median of vdisp_mu
            tau = self.params[self.freepar_idx["tau"]]
            self.params[self.freepar_idx["vdisp_mu"]] = vdisp[0]
            self.params[self.freepar_idx["vdisp"]] = (
                np.diff(vdisp) / (tau * self._walk_weight) if tau > 0.00 else 0.00
            )

        self.bbobj.params = self._physical_theta(self.params)[: len(self.bbobj.freepar_names)]
        self.bbobj.modCalculated = True

        if self._fit_rsep:
            self.bbobj.rsep = float(self.params[self.freepar_idx["rsep"]])
            self.bbobj.radii = self._radii(self.bbobj.rsep)

        self.log_l_median = self._log_likelihood(self.params)
        print(
            f"Log-likelihood at the median parameters: {self.log_l_median:.2f}; "
            f"maximum over the samples: {self.log_l.max():.2f}"
        )

    # -----------------------------------------------------------------------------
    # - Collapse z-space vdisp samples to physical per-ring vdisp for reporting
    # -----------------------------------------------------------------------------
    def _display_samples(self, use_physical_params=True):
        if not self._vdisp_constrained:
            if not use_physical_params:
                raise ValueError(
                    "use_physical_params=False requires the 'constrained' vdisp prior mode."
                )
            return self.samples, self.freepar_names

        if not use_physical_params:
            return self.samples, self.freepar_names

        # Physical layout; vdisp_mu is dropped since it equals vdisp1
        samples = self._physical_theta(self.samples)
        keep = [i for i in range(len(self._phys_names)) if i != self._phys_idx["vdisp_mu"]]
        return samples[:, keep], [self._phys_names[i] for i in keep]

    # -----------------------------------------------------------------------------
    # - Save corner plot
    # -----------------------------------------------------------------------------
    def save_corner(self, sigma=10.00, use_physical_params=True, **kwargs):
        samples, names = self._display_samples(use_physical_params=use_physical_params)

        if sigma is not None:
            edges = np.array(
                [
                    corner.quantile(s, [0.16, 0.50, 0.84], weights=self.weights)
                    for s in samples.T
                ]
            )
            edges = np.array(
                [
                    [
                        np.maximum(
                            e[1] - sigma * (e[1] - e[0]), samples[:, i].min()
                        ),
                        np.minimum(
                            e[1] + sigma * (e[2] - e[1]), samples[:, i].max()
                        ),
                    ]
                    for i, e in enumerate(edges)
                ]
            )
        else:
            edges = np.array([[s.min(), s.max()] for s in samples.T])

        show_titles = kwargs.pop("show_titles", True)

        # corner.corner forwards unnamed options to hist2d and ignores unknown ones
        internal = {"data", "xs", "x", "y", "weights", "labels", "range", "ax", "new_fig"}
        allowed = set(inspect.signature(corner.corner).parameters) | set(inspect.signature(corner.hist2d).parameters)
        allowed -= internal | {"kwargs", "hist2d_kwargs"}
        if set(kwargs) & internal:
            raise TypeError(f"save_corner() sets {sorted(set(kwargs) & internal)} internally")
        unknown = sorted(set(kwargs) - allowed)
        if unknown:
            raise TypeError(f"Unrecognized arguments for save_corner(): {unknown}")
        corner_kwargs = kwargs

        corner.corner(
            samples,
            weights=self.weights,
            labels=names,
            show_titles=show_titles,
            range=edges,
            **corner_kwargs,
        )

        plt.savefig(f"{self.output}_corner.pdf", format="pdf", dpi=300)
        plt.close()


    # -----------------------------------------------------------------------------
    # - Save 16-50-84 percentiles for each free parameter
    # -----------------------------------------------------------------------------
    def save_percentiles(self, use_physical_params=True):
        samples, names = self._display_samples(use_physical_params=use_physical_params)

        edges = np.array(
            [
                corner.quantile(s, [0.16, 0.50, 0.84], weights=self.weights)
                for s in samples.T
            ]
        )

        header = f"{'param':<20} {'p16':>15} {'p50':>15} {'p84':>15}"
        rows = [
            f"{name:<20} {p16:>15.6e} {p50:>15.6e} {p84:>15.6e}"
            for name, (p16, p50, p84) in zip(names, edges)
        ]

        with open(f"{self.output}_percentiles.txt", "w") as f:
            f.write("\n".join([header] + rows) + "\n")

        return edges


    # -----------------------------------------------------------------------------
    # - Save best-fit model and outputs using the best-fit parameters
    # -----------------------------------------------------------------------------
    def save_best_model(self, plots=True, which="median", **kwargs):
        if not self.bbobj.modCalculated:
            print(
                "Sampler has not been run yet. Please run compute() before running this function."
            )
            return

        choices = {"median": "params", "maxlike": "params_maxlike", "maxpost": "params_maxpost"}
        if which not in choices:
            raise ValueError(f"Invalid value for 'which': {which!r}. Please use one of: {list(choices)}")
        theta = getattr(self, choices[which])

        model_out = self._get_model(theta, convolve=True)
        if not isinstance(model_out, tuple):
            raise ValueError(
                "The best-fit parameters fall outside the allowed range: no model can be built."
            )
        model = model_out[0]

        rings = self.bbobj._update_rings(self.bbobj._inri, self._physical_theta(theta))

        if self.bbobj.update_prof:
            self.bbobj._update_profile(rings)

        # BBarolo's PV plot scripts fail on a ring at r=0: leave it out of the output rings
        outri = rings
        if rings.r["radii"][0] == 0.00:
            outri = Rings(rings.nr - 1)
            outri.set_rings_from_dict({key: values[1:] for key, values in rings.r.items()})
        libBB.Galfit_setOutRings(self.bbobj._galfit, outri._rings)

        _, _, _, galmod = self.bbobj._render_model(rings, fullcube=True)

        buffer = reshapePointer(libBB.Galmod_array(galmod), model.shape)
        buffer[:] = model

        libBB.Galfit_writeOutputs(
            self.bbobj._galfit, galmod, self.bbobj._ellprof, plots
        )
        libBB.Galmod_delete(galmod)
        plt.close()

        # Leave the ring grid at the median, as bbobj.rsep and bbobj.radii
        if self._fit_rsep:
            self.bbobj._inri.modify_parameter("radii", self.bbobj.radii, makeobj=True)