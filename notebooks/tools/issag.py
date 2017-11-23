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
                 domain_t_trun=(0.0, 13.7e9),
                 log_domain_tau_trun=(7.0, 9.0),
                 domain_t_burst=(0.0, 13.7e9),
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

        self.t_trun = _random_range_(self.domain_t_trun)
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

        min_t_ext = min(self.domain_t_ext[0], self.t_burst)
        max_t_ext = min(self.domain_t_ext[1], self.t_burst)
        self.t_ext = _random_range_((min_t_ext, max_t_ext))
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
        if not path:
            self.path = os.path.expandvars("#{ssp-bc03}".format(env_var))
        else:
            if not os.path.exists(path):
                raise(ValueError, "path '{path}' doesn't exist.".format(path))
            self.path = path

        if not match:
            self.match = "*"
        else:
            self.match = match

        self.models_list = sorted([os.path.join(root, file)
                                  for root, subs, files in os.walk(self.path)
                                  for file in files if fnmatch(file,
                                                               self.match)])

    def get_label(self, fits_object):
        return fits_object[2].data["Zstars"][0]

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
                                columns=self.get_pegase_ages(fits_object))
        SEDs_tot = pd.DataFrame(fl_ste_tot+fl_neb_tot, index=wl_sor,
                                columns=self.get_pegase_ages(fits_object))

        return SEDs_ste, SEDs_tot

    def set_all_models(self):
        """Reads ALL models found in path."""

        labels, ages, SEDs_stellar, SEDs_nebular = [], [], [], []
        for fits_name in self.models_list:
            with fits.open(fits_name) as fits_object:
                models = self.read_pegase_model(fits_object)
                labels += [self.get_pegase_label(fits_object)]
                SEDs_stellar += [models[1]]
                SEDs_nebular += [models[2]]
                ages += [self.get_pegase_ages(fits_object)]
        if not all([all(ages[0] == ages_i) for ages_i in ages[1:]]):
            raise(ValueError, "not all models have same age sampling.")

        self.ages = ages[0]
        self.SEDs_stellar = OrderedDict(zip(labels, SEDs_stellar))
        self.SEDs_nebular = OrderedDict(zip(labels, SEDs_nebular))

        return None


class iSSAG(object):

    def __init__(self, size=1000):
        self.chen = Sampler()
        self.sample = self.chen.get_samples(size)

        self.models = Models()
        self.SFHs = self.set_all_SFHs()

    def get_timescale(self, iloc):
        """Define timescale for the iloc SFH."""
        # models time scale
        t = self.models.ages.values
        # SSAG time parameters
        ssag_time = sorted([self.sample.t_form[iloc],
                            self.sample.t_burst[iloc],
                            self.sample.t_trun[iloc]])
        # build timescale
        for i in xrange(len(ssag_time)):
            if ssag_time[i] not in t:
                t = np.insert(t, np.searchsorted(t, ssag_time[i]),
                              ssag_time[i])
        return t

    def t_interpolate(self, iloc):
        """Interpolate models in time."""
        pass

    def Z_interpolate(self, iloc):
        """Interpolate models in metallicity."""
        pass

    def get_SFH_cont(self, iloc):
        """Build continuous part of the SFH."""
        pass

    def get_SFH_burst(self, iloc):
        """Build burst part of the SFH."""
        pass

    def get_SFH_trun(self, iloc):
        """Build truncation part of the SFH."""
        pass

    def get_SFH(self, parameters):

        SFH = pd.Series()
        return SFH

    def set_all_SFHs(self):
        pass
