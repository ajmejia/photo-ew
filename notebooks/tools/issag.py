# Idea original from Ivan Cabrera-Ziri (~2014)
# This code will compute the SSAG SFHs
#

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
from collections import OrderedDict
from astropy.io import fits
from fnmatch import fnmatch


def _random_range_(domain):
    return domain[0] + np.random.rand() * (domain[1] - domain[0])


def _rejection_(pdf, domain, top, size=1):
    N = 0
    dev = []
    while N < size:
        rx = _random_range_(domain)
        ru = np.random.rand()
        if ru < pdf(rx)/top:
            dev += [rx]
            N += 1
    if size == 1:
        return dev[0]
    else:
        return dev


class Sampler(object):

    def __init__(self, domain_t_form=(1.5e9, 13.7e9),
                 domain_gamma=(0.0, 1.0),
                 domain_t_trun=(1e6, 13.7e9),
                 log_domain_tau_trun=(7.0, 9.0),
                 domain_t_burst=(1e6+3e8, 13.7e9),
                 domain_t_ext=(3e7, 3e8),
                 domain_A=(0.03, 4.0),
                 domain_Z=(0.005, 2.5),
                 domain_tau_V=(0.0, 6.0),
                 domain_mu_V=(0.0, 1.0),
                 domain_sigma_v=(50.0, 400.0)):
        self.domain_t_form = domain_t_form
        self.domain_gamma = domain_gamma
        self.domain_t_trun = domain_t_trun
        self.log_domain_tau_trun = log_domain_tau_trun
        self.domain_t_burst = domain_t_burst
        self.domain_t_ext = domain_t_ext
        self.domain_A = domain_A
        self.domain_Z = domain_Z
        self.domain_tau_V = domain_tau_V
        self.domain_mu_V = domain_mu_V
        self.domain_sigma_v = domain_sigma_v

        self.t_form = None
        self.gamma = None
        self.truncated = None
        self.t_trun = None
        self.tau_trun = None
        self.t_burst = None
        self.t_ext = None
        self.A = None
        self.Z = None
        self.tau_V = None
        self.mu_V = None
        self.sigma_v = None

        self.minimum_sample_size = 10000
        self.sample = None

        self.truncated_swap_counter = 0

    def _clean_draws_(self):
        self.t_form = None
        self.gamma = None
        self.truncated = None
        self.t_trun = None
        self.tau_trun = None
        self.t_burst = None
        self.t_ext = None
        self.A = None
        self.Z = None
        self.tau_V = None
        self.mu_V = None
        self.sigma_v = None
        return None

    def draw_t_form(self):
        self.t_form = _random_range_(self.domain_t_form)
        return self.t_form

    def draw_gamma(self):
        self.gamma = _random_range_(self.domain_gamma)
        return self.gamma

    def draw_truncated(self):
        if self.gamma is None:
            self.draw_gamma()

        self.truncated = True if np.random.rand() < 0.3 else False

        if self.truncated and\
                1.0/self.gamma*1e9 < 10**self.log_domain_tau_trun[0]:
            self.truncated_swap_counter += 1
            self.truncated = False
            return self.truncated
        elif not self.truncated and self.truncated_swap_counter > 0:
            self.truncated_swap_counter -= 1
            self.truncated = True
            return self.truncated

        return self.truncated

    def draw_t_trun(self):
        if self.truncated is None:
            self.draw_truncated()
        if self.truncated is False:
            self.t_trun = np.nan
            return self.t_trun
        if not self.t_form:
            self.draw_t_form()

        self.t_trun = _random_range_((self.domain_t_trun[0], self.t_form))
        return self.t_trun

    def draw_tau_trun(self):
        if self.truncated is None:
            self.draw_truncated()
        if self.truncated is False:
            self.tau_trun = np.nan
            return self.tau_trun
        if self.gamma is None:
            self.draw_gamma()

        max_tau_trun = min(10**self.log_domain_tau_trun[1], 1.0/self.gamma*1e9)
        self.tau_trun = 10**_random_range_((self.log_domain_tau_trun[0],
                                            np.log10(max_tau_trun)))
        return self.tau_trun

    def draw_t_burst(self):
        if self.t_form is None:
            self.draw_t_form()

        tol_t_burst = 0.10 if self.truncated else 0.15
        domain_t_burst = (self.domain_t_burst[0], 2e9)\
            if np.random.rand() <= tol_t_burst\
            else (2e9, self.t_form)
        self.t_burst = _random_range_(domain_t_burst)
        return self.t_burst

    def draw_t_ext(self):
        if not self.t_burst:
            self.draw_t_burst()

        self.t_ext = _random_range_(self.domain_t_ext)
        return self.t_ext

    def draw_A(self):
        self.A = 10**_random_range_((np.log10(self.domain_A[0]),
                                     np.log10(self.domain_A[1])))
        return self.A

    def draw_Z(self):
        domain_Z = (self.domain_Z[0], 0.2) if np.random.rand() <= 0.05 \
                    else (0.2, self.domain_Z[1])
        self.Z = _random_range_(domain_Z)
        return self.Z

    def draw_tau_V(self, mean=1.2, std=0.9851185):
        self.tau_V = _rejection_(lambda x: norm.pdf(x, loc=mean, scale=std),
                                 domain=self.domain_tau_V,
                                 top=norm.pdf(mean, loc=mean, scale=std))
        return self.tau_V

    def draw_mu_V(self, mean=0.3, std=0.36573657):
        self.mu_V = _rejection_(lambda x: norm.pdf(x, loc=mean, scale=std),
                                domain=self.domain_mu_V,
                                top=norm.pdf(mean, loc=mean, scale=std))
        return self.mu_V

    def draw_sigma_v(self):
        self.sigma_v = _random_range_(self.domain_sigma_v)
        return self.sigma_v

    def get_samples(self, size=1, pristine=False):
        self._clean_draws_()

        columns = ["t_form", "gamma", "truncated", "t_trun", "tau_trun",
                   "t_burst", "t_ext", "A", "Z", "tau_V", "mu_V", "sigma_v"]
        sample = OrderedDict([(kw, []) for kw in columns])
        for i in xrange(size):
            sample["t_form"] += [self.draw_t_form()]
            sample["gamma"] += [self.draw_gamma()]
            sample["truncated"] += [self.draw_truncated()]
            sample["t_trun"] += [self.draw_t_trun()]
            sample["tau_trun"] += [self.draw_tau_trun()]
            sample["t_burst"] += [self.draw_t_burst()]
            sample["t_ext"] += [self.draw_t_ext()]
            sample["A"] += [self.draw_A()]
            sample["Z"] += [self.draw_Z()]
            sample["tau_V"] += [self.draw_tau_V()]
            sample["mu_V"] += [self.draw_mu_V()]
            sample["sigma_v"] += [self.draw_sigma_v()]
        if pristine or self.sample is None:
            self.sample = pd.DataFrame(sample, columns=columns)
        else:
            self.sample = self.sample.append(pd.DataFrame(sample,
                                             columns=columns),
                                             ignore_index=True)
        return self.sample


class Models(object):

    def __init__(self, env_var="ssp-bc03", path=None, match=None):
        """Load SSP models from given path OR environment variable."""
        if path is None:
            path = os.path.expandvars("${}".format(env_var))
        else:
            if not os.path.exists(path):
                raise ValueError("path '{}' doesn't exist.".format(path))

        if not match:
            match = "*"

        self.models_list = sorted([os.path.join(root, file)
                                  for root, subs, files in os.walk(path)
                                  for file in files
                                  if fnmatch(file, match)])
        self.metallicities = None
        self.ages = None
        self.wavelenth = None
        self.SEDs_stellar = None
        self.SEDs_nebular = None

    def get_metallicities(self, fits_object):
        return fits_object[2].data["Zstars"][0] / 0.02

    def get_ages(self, fits_object):
        return fits_object[2].data["AGE"]*1e6

    def get_SSP(self, fits_object):
        wl_ste = fits_object[3].data["BFIT"]
        fl_ste = fits_object[0].data.T
        wl_neb = fits_object[1].data["WAVE"]
        fl_neb = fits_object[1].data["FLUXLINE"]

        wl_tot = np.concatenate((wl_ste, wl_neb))
        wl_sor, uniq_wl = np.unique(wl_tot, return_index=True)

        fl_ste_tot = np.zeros((uniq_wl.size, fl_ste.shape[1]))
        for j in xrange(fl_ste_tot.shape[1]):
            fl_ste_tot[:, j] = np.interp(wl_sor, wl_ste, fl_ste[:, j],
                                         left=0.0, right=0.0)

        fl_neb_tot = np.zeros((wl_tot.size, fl_ste.shape[1]))
        fl_neb_tot[-wl_neb.size:] = fl_neb
        fl_neb_tot = fl_neb_tot[uniq_wl]

        SEDs_ste = pd.DataFrame(fl_ste_tot, index=wl_sor,
                                columns=self.get_ages(fits_object))
        SEDs_tot = pd.DataFrame(fl_ste_tot+fl_neb_tot, index=wl_sor,
                                columns=self.get_ages(fits_object))

        return SEDs_ste, SEDs_tot

    def set_all(self):
        """Reads ALL models found in path."""

        metallicities, ages, SEDs_stellar, SEDs_nebular = [], [], [], []
        for fits_name in self.models_list:
            with fits.open(fits_name) as fits_object:
                stellar, nebular = self.get_SSP(fits_object)
                metallicities += [self.get_metallicities(fits_object)]
                SEDs_stellar += [stellar]
                SEDs_nebular += [nebular]
                ages += [self.get_ages(fits_object)]
        if not all([all(ages[0] == ages_i) for ages_i in ages[1:]]):
            raise(ValueError, "not all models have same age sampling.")

        self.metalicities = pd.Series(metallicities)
        self.ages = ages[0]
        self.wavelength = stellar.index.values
        self.SEDs_stellar = OrderedDict(zip(metallicities, SEDs_stellar))
        self.SEDs_nebular = OrderedDict(zip(metallicities, SEDs_nebular))

        return None


class iSSAG(object):
    """A Star Formation History (SFH) library generator with
       parametric approach a la Chen+2012.

       Attributes:
           chen     The sampler used to draw SFH parameters.
           sample   The samples drawn for the current library.
           models   The models loader.
           SFHs     The SFH library.
       """

    def __init__(self, size=1000,
                 path="/home/mejia/Research/photometric-ew/models/PEGASE/"):
        # initialize SFH sampler
        self.chen = Sampler()
        # draw some samples
        self.sample = self.chen.get_samples(size)
        # initialize models loader
        self.models = Models(path=path)
        # read all models in default path
        self.models.set_all()
        # initial value for library
        self.SFHs = None

    def get_metallicity_interpolation(self, iloc):
        """Interpolate models in metallicity."""
        Z = self.models.metalicities.values.copy()
        Z_new = self.sample.Z[iloc]
        models = self.models.SEDs_nebular.copy()
        if Z_new not in Z:
            j = np.searchsorted(Z, Z_new)
            Z_0, Z_1, Z_2 = np.log10([Z_new, Z[j-1], Z[j]])
            v, w = (Z_2 - Z_0)/(Z_2 - Z_1), (Z_0 - Z_1)/(Z_2 - Z_1)
            new_model = v * models.get(Z[j-1]) + w * models.get(Z[j])

        else:
            new_model = models.get(Z_new)
        return new_model

    def get_time_interpolation(self, iloc, SSP):
        """Interpolate models in time."""
        # copy models timescale
        t = SSP.columns.copy()
        # SSAG time parameters
        t_new = [self.sample.t_form[iloc],
                 self.sample.t_burst[iloc],
                 self.sample.t_burst[iloc] - self.sample.t_ext[iloc]]
        if self.sample.truncated[iloc]:
            t_new += [self.sample.t_trun[iloc]]
        for i in xrange(len(t_new)):
            t = SSP.columns.copy()
            if t_new[i] not in t:
                # find column j such that:
                # t[j] < t_new[i] < t[j+1]
                j = np.searchsorted(t, t_new[i])
                # interpolate in models (j-1, j) assuming linearity in log-ages

                t_0, t_1, t_2 = np.log10([t_new[i], t[j-1], t[j]])
                v, w = (t_2 - t_0)/(t_2 - t_1), (t_0 - t_1)/(t_2 - t_1)
                new_model = v * SSP.get(t[j-1]) + w * SSP.get(t[j])
                SSP.insert(j, t_new[i], new_model)
        return SSP

    def get_SFH(self, iloc, timescale):
        """Build SFH from iloc galaxy in the sample."""
        t_form = self.sample.t_form[iloc]
        gamma = self.sample.gamma[iloc]
        t_burst_i = self.sample.t_burst[iloc]
        t_burst_f = t_burst_i - self.sample.t_ext[iloc]
        A = self.sample.A[iloc]
        t_trun = self.sample.t_trun[iloc]
        tau_trun = self.sample.tau_trun[iloc]

        if not all([time in timescale for time in [t_form,
                                                   t_burst_i,
                                                   t_burst_f,
                                                   t_trun] if pd.notna(time)]):
            raise ValueError("You need to interpolate in time first.")

        # build truncation (SFH_trun) and main SFH (SFH_cont)
        mask_cont = np.ones(timescale.size, dtype=np.bool)
        mask_cont[timescale > t_form] = False
        SFH_trun = np.zeros(timescale.size)
        if self.sample.truncated[iloc]:
            mask_cont[timescale <= t_trun] = False
            mask_trun = np.ones(timescale.size, dtype=np.bool)
            mask_trun[timescale > t_trun] = False
            SFH_trun[mask_trun] = np.exp(-(t_trun - timescale[mask_trun]) /
                                         tau_trun)
            SFH_trun[mask_trun] *= np.exp(-(t_form - t_trun) * gamma*1e-9)
        SFH_cont = np.zeros(timescale.size)
        SFH_cont[mask_cont] = np.exp(-(t_form - timescale[mask_cont]) *
                                     gamma*1e-9)
        # build burst (SFH_burst)
        mask_burst = np.ones(timescale.size, dtype=np.bool)
        mask_burst[t_burst_f > timescale] = False
        mask_burst[t_burst_i < timescale] = False
        mass_under = np.trapz(SFH_cont+SFH_trun, timescale)
        SFH_burst = np.zeros(timescale.size)
        SFH_burst[mask_burst] = mass_under * A / (t_burst_i - t_burst_f)

        SFH = pd.Series(SFH_cont+SFH_trun+SFH_burst, timescale, name=iloc)

        return SFH

    def set_all_SFHs(self):
        """Build SFH library."""
        SFHs = OrderedDict()
        for i in self.sample.index:
            SSP = self.get_metallicity_interpolation(i)
            SSP = self.get_time_interpolation(i, SSP)

            SFHs[i] = self.get_SFH(i, SSP.columns)

        self.SFHs = SFHs

        return None

    def set_all_SEDs(self):
        """Build both: the SED and the SFHs."""
        SFHs, SEDs = OrderedDict(), OrderedDict()
        for i in self.sample.index:
            SSP = self.get_metallicity_interpolation(i)
            SSP = self.get_time_interpolation(i, SSP)

            SFHs[i] = self.get_SFH(i, SSP.columns)
            SEDs[i] = np.average(SSP.values,
                                 weights=np.tile(SFHs[i],
                                                 (SSP.index.size, 1)),
                                 axis=1)

        self.SFHs = SFHs
        self.SEDs = pd.DataFrame(SEDs, index=self.models.wavelength)

        return None
