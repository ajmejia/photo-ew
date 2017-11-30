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
                 domain_t_ext=(3e7, 3e8),
                 domain_t_burst=(1e6, 13.7e9),
                 domain_a_burst=(0.03, 4.0),
                 domain_z=(0.005, 2.5),
                 domain_tau_v=(0.0, 6.0),
                 domain_mu_v=(0.0, 1.0),
                 domain_sigma_v=(50.0, 400.0)):
        self.domain_t_form = domain_t_form
        self.domain_gamma = domain_gamma
        self.domain_t_trun = domain_t_trun
        self.log_domain_tau_trun = log_domain_tau_trun
        self.domain_t_burst = domain_t_burst
        self.domain_t_ext = domain_t_ext
        self.domain_a_burst = domain_a_burst
        self.domain_z = domain_z
        self.domain_tau_v = domain_tau_v
        self.domain_mu_v = domain_mu_v
        self.domain_sigma_v = domain_sigma_v

        self.t_form = None
        self.gamma = None
        self.truncated = None
        self.t_trun = None
        self.tau_trun = None
        self.t_burst = None
        self.t_ext = None
        self.a_burst = None
        self.metallicity = None
        self.tau_v = None
        self.mu_v = None
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
        self.a_burst = None
        self.metallicity = None
        self.tau_v = None
        self.mu_v = None
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

    def draw_t_ext(self):
        self.t_ext = _random_range_(self.domain_t_ext)
        return self.t_ext

    def draw_t_burst(self):
        if self.t_form is None:
            self.draw_t_form()
        if self.t_ext is None:
            self.draw_t_ext()

        # tol_t_burst = 0.10 if self.truncated else 0.15
        domain_t_burst = (self.domain_t_burst[0]+self.t_ext,
                          (2e9 if self.t_form >= 2e9 else self.t_form))\
            if np.random.rand() < 0.1 else (2e9, self.t_form)
        self.t_burst = _random_range_(domain_t_burst)
        return self.t_burst

    def draw_a_burst(self):
        self.a_burst = 10**_random_range_((np.log10(self.domain_a_burst[0]),
                                           np.log10(self.domain_a_burst[1])))
        return self.a_burst

    def draw_z(self):
        domain_z = (self.domain_z[0], 0.2) if np.random.rand() <= 0.05 \
                    else (0.2, self.domain_z[1])
        self.metallicity = _random_range_(domain_z)
        return self.metallicity

    def draw_tau_v(self, mean=1.2, std=0.9851185):
        self.tau_v = _rejection_(lambda x: norm.pdf(x, loc=mean, scale=std),
                                 domain=self.domain_tau_v,
                                 top=norm.pdf(mean, loc=mean, scale=std))
        return self.tau_v

    def draw_mu_v(self, mean=0.3, std=0.36573657):
        self.mu_v = _rejection_(lambda x: norm.pdf(x, loc=mean, scale=std),
                                domain=self.domain_mu_v,
                                top=norm.pdf(mean, loc=mean, scale=std))
        return self.mu_v

    def draw_sigma_v(self):
        self.sigma_v = _random_range_(self.domain_sigma_v)
        return self.sigma_v

    def get_samples(self, size=1, pristine=False):
        self._clean_draws_()

        columns = ["t_form", "gamma", "truncated", "t_trun", "tau_trun",
                   "t_burst", "t_ext", "a_burst", "metallicity", "tau_v",
                   "mu_v", "sigma_v"]
        sample = OrderedDict([(kw, []) for kw in columns])
        for i in xrange(size):
            sample["t_form"] += [self.draw_t_form()]
            sample["gamma"] += [self.draw_gamma()]
            sample["truncated"] += [self.draw_truncated()]
            sample["t_trun"] += [self.draw_t_trun()]
            sample["tau_trun"] += [self.draw_tau_trun()]
            sample["t_ext"] += [self.draw_t_ext()]
            sample["t_burst"] += [self.draw_t_burst()]
            sample["a_burst"] += [self.draw_a_burst()]
            sample["metallicity"] += [self.draw_z()]
            sample["tau_v"] += [self.draw_tau_v()]
            sample["mu_v"] += [self.draw_mu_v()]
            sample["sigma_v"] += [self.draw_sigma_v()]
            self._clean_draws_()
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
        self.wavelength = None
        self.ssps_stellar = None
        self.ssps_nebular = None

    def get_metallicities(self, fits_object):
        return fits_object[2].data["Zstars"][0] / 0.02

    def get_ages(self, fits_object):
        return fits_object[2].data["AGE"]*1e6

    def get_ssp(self, fits_object):
        wl_ste = fits_object[3].data["BFIT"]
        fl_ste = fits_object[0].data.T
        wl_neb = fits_object[1].data["WAVE"]
        fl_neb = fits_object[1].data["FLUXLINE"]

        wl_merged = np.concatenate((wl_ste, wl_neb))
        wl_sor, uniq_wl = np.unique(wl_merged, return_index=True)

        fl_ste_merged = np.zeros((uniq_wl.size, fl_ste.shape[1]))
        for j in xrange(fl_ste_merged.shape[1]):
            fl_ste_merged[:, j] = np.interp(wl_sor, wl_ste, fl_ste[:, j],
                                            left=0.0, right=0.0)

        fl_neb_merged = np.zeros((wl_merged.size, fl_ste.shape[1]))
        fl_neb_merged[-wl_neb.size:] = fl_neb
        fl_neb_merged = fl_neb_merged[uniq_wl]

        ssp_stellar = pd.DataFrame(fl_ste_merged, index=wl_sor,
                                   columns=self.get_ages(fits_object))
        ssp_nebular = pd.DataFrame(fl_ste_merged+fl_neb_merged, index=wl_sor,
                                   columns=self.get_ages(fits_object))

        return ssp_stellar, ssp_nebular

    def set_all(self):
        """Reads ALL models found in path."""

        metallicities, ages, ssps_stellar, ssps_nebular = [], [], [], []
        for fits_name in self.models_list:
            with fits.open(fits_name) as fits_object:
                stellar, nebular = self.get_ssp(fits_object)
                metallicities += [self.get_metallicities(fits_object)]
                ssps_stellar += [stellar]
                ssps_nebular += [nebular]
                ages += [self.get_ages(fits_object)]
        if not all([all(ages[0] == ages_i) for ages_i in ages[1:]]):
            raise(ValueError, "not all models have same age sampling.")

        self.metallicities = pd.Series(metallicities)
        self.ages = ages[0]
        self.wavelength = stellar.index.values
        self.ssps_stellar = OrderedDict(zip(metallicities, ssps_stellar))
        self.ssps_nebular = OrderedDict(zip(metallicities, ssps_nebular))

        return None


class iSSAG(object):
    """A Star Formation History (SFH) library generator with
       parametric approach a la Chen et al. (2012).

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
        self.sfhs = None
        # initial value for SEDs
        self.seds_nebular = None
        self.seds_stellar = None

    def get_extinction_curve(self, iloc, timescale):
        """Build extinction curve from Charlot & Fall (2001)."""
        wl = self.models.wavelength
        tau_v = self.sample.tau_v[iloc]
        mu_v = self.sample.mu_v[iloc]
        n_BC = np.count_nonzero(timescale < 1e7)
        n_ISM = timescale.size - n_BC

        ext_BC = np.tile(np.exp(-tau_v*(wl/5500.0)**(-0.7)),
                         (n_BC, 1)).T
        ext_ISM = np.tile(np.exp(-mu_v*tau_v*(wl/5500.0)**(-0.7)),
                          (n_ISM, 1)).T

        return np.column_stack((ext_BC, ext_ISM))

    def get_kinematic_effects(self, iloc, sed):
        """Adds velocity dispersion to given SED.
           This method was adapted from GALAXEV routines
           (Bruzual & Charlot, 2003).
           """
        c_mks = 3e5
        m = 6.0
        nx = 100
        wl = self.models.wavelength
        nwl = wl.size + 2*nx
        # losvd = self.sample.sigma_v[iloc]
        losvd = 400.0

        wl_, sed_ = np.zeros(nwl), np.zeros(nwl)
        wl_[:nx] = wl[0]
        wl_[nx:nwl-nx] = wl
        wl_[nwl-nx:] = wl[-1]
        sed_[:nx] = sed[0]
        sed_[nx:nwl-nx] = sed
        sed_[nwl-nx:] = sed[-1]
        for i in xrange(nwl):
            wl_max = c_mks*wl_[i] / (c_mks-m*losvd)
            j = np.searchsorted(wl_, wl_max)
            m2 = j + 1
            m1 = 2 * i - m2

            if m1 < 0:
                m1 = 0
            if m2 > nwl:
                m2 = nwl

            u, g = [], []
            for j in xrange(m2-1, m1-1, -1):
                u += [(wl_[i] / wl_[j]-1.0) * c_mks]
                g += [sed_[j] * norm.pdf(u[-1], loc=0.0, scale=losvd)]

            if i >= nx+1 and i < nwl-nx:
                sed_[i-nx] = np.trapz(g, x=u)

        return sed_[nx:-nx]

    def get_metallicity_interpolation(self, iloc, emission="nebular"):
        """Interpolate models in metallicity."""
        zs = self.models.metallicities.values.copy()
        z_new = self.sample.metallicity[iloc]
        if emission == "nebular":
            ssps = (self.models.ssps_nebular.copy(),)
        elif emission == "stellar":
            ssps = (self.models.ssps_stellar.copy(),)
        elif emission == "both":
            ssps = (self.models.ssps_nebular.copy(),
                    self.models.ssps_stellar.copy())
        else:
            raise ValueError("Invalid model. Try: 'nebular', \
                              'stellar' or 'both'.")
        ssps_new = []
        if z_new not in zs:
            for ssp in ssps:
                j = np.searchsorted(zs, z_new)
                z_0, z_1, z_2 = np.log10([z_new, zs[j-1], zs[j]])
                v, w = (z_2 - z_0)/(z_2 - z_1), (z_0 - z_1)/(z_2 - z_1)
                ssps_new += [v * ssp.get(zs[j-1]) + w * ssp.get(zs[j])]
        else:
            ssps_new += [ssp.get(z_new) for ssp in ssps]
        return ssps_new

    def get_time_interpolation(self, iloc, ssps):
        """Interpolate models in time."""
        # SSAG time parameters
        t_new = [self.sample.t_form[iloc],
                 self.sample.t_burst[iloc],
                 self.sample.t_burst[iloc] - self.sample.t_ext[iloc]]
        if self.sample.truncated[iloc]:
            t_new += [self.sample.t_trun[iloc]]
        for i in xrange(len(t_new)):
            t = ssps[0].columns.copy()
            if t_new[i] not in t:
                # find column j such that:
                # t[j] < t_new[i] < t[j+1]
                k = np.searchsorted(t, t_new[i])
                # interpolate in models (k-1, k) assuming linearity in log-ages

                t_0, t_1, t_2 = np.log10([t_new[i], t[k-1], t[k]])
                v, w = (t_2 - t_0)/(t_2 - t_1), (t_0 - t_1)/(t_2 - t_1)
                for j in xrange(len(ssps)):
                    new_model = v * ssps[j].get(t[k-1]) + w * ssps[j].get(t[k])
                    ssps[j].insert(k, t_new[i], new_model)
        return ssps

    def get_sfh(self, iloc, timescale):
        """Build SFH from iloc galaxy in the sample."""
        t_form = self.sample.t_form[iloc]
        gamma = self.sample.gamma[iloc]
        t_burst_i = self.sample.t_burst[iloc]
        t_burst_f = t_burst_i - self.sample.t_ext[iloc]
        a_burst = self.sample.a_burst[iloc]
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
        SFH_burst[mask_burst] = mass_under * a_burst / (t_burst_i - t_burst_f)

        SFH = pd.Series(SFH_cont+SFH_trun+SFH_burst, timescale, name=iloc)

        return SFH

    def set_all_sfhs(self):
        """Build SFH library."""
        sfhs = OrderedDict()
        for i in self.sample.index:
            ssps = self.get_metallicity_interpolation(i)
            ssps = self.get_time_interpolation(i, ssps)

            sfhs[i] = self.get_sfh(i, ssps[0].columns)

        self.sfhs = sfhs

        return None

    def set_all_seds(self, emission="nebular"):
        """Build both: the SED and the SFHs."""
        sfhs = OrderedDict()
        seds = []
        columns = self.sample.index
        for i in columns:
            ssps = self.get_metallicity_interpolation(i, emission=emission)
            ssps = self.get_time_interpolation(i, ssps)
            sfhs[i] = self.get_sfh(i, ssps[0].columns)

            for j in xrange(len(ssps)):
                ssps[j] *= self.get_extinction_curve(i, ssps[j].columns)

                seds += [np.average(ssps[j].values,
                                    weights=np.tile(sfhs[i],
                                                    (ssps[j].index.size, 1)),
                                    axis=1)]
                # seds[-1] = self.get_kinematic_effects(i, seds[-1])

        self.sfhs = sfhs
        if emission == "both":
            self.seds_nebular = pd.DataFrame(np.array(seds[::2]).T,
                                             index=self.models.wavelength,
                                             columns=columns)
            self.seds_stellar = pd.DataFrame(np.array(seds[1::2]).T,
                                             index=self.models.wavelength,
                                             columns=columns)
        elif emission == "nebular":
            self.seds_nebular = pd.DataFrame(np.array(seds).T,
                                             index=self.models.wavelength,
                                             columns=columns)
        elif emission == "stellar":
            self.seds_stellar = pd.DataFrame(np.array(seds).T,
                                             index=self.models.wavelength,
                                             columns=columns)
        return None
