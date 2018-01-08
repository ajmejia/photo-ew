# Idea original from Ivan Cabrera-Ziri (~2014)
# Implementation by Alfredo Mejia-Narvaez (2017)
# This code will compute the SSAG SFHs
#

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
from collections import OrderedDict
from tools.Photometry import integrated_flux as iflux
from astropy.io import fits
from fnmatch import fnmatch
from astropy import cosmology
import astropy.units as u


UNIVERSE = cosmology.FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc,
                                   Tcmb0=2.725 * u.K, Om0=0.3)
UNIVERSE_AGE = UNIVERSE.age(0.0).to(u.yr).value
FIRST_GAL_AGE = UNIVERSE_AGE - 1e6


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

    def __init__(self, domain_t_form=(1.5e9, FIRST_GAL_AGE),
                 domain_gamma=(0.0, 1.0),
                 domain_t_trun=(1e6, FIRST_GAL_AGE),
                 log_domain_tau_trun=(7.0, 9.0),
                 domain_t_ext=(3e7, 3e8),
                 domain_t_burst=(1e6, FIRST_GAL_AGE),
                 domain_a_burst=(0.03, 4.0),
                 domain_z=(0.005, 2.5),
                 domain_tau_v=(0.0, 6.0),
                 domain_mu_v=(0.0, 1.0),
                 domain_sigma_v=(50.0, 400.0),
                 domain_redshift=(0.0, 1000.0)):
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
        self.domain_redshift = domain_redshift

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
        self.redshift = None
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

        tol_t_burst = 0.10 if self.truncated else 0.15
        if self.t_form >= 2e9:
            domain_t_burst = (self.domain_t_burst[0]+self.t_ext, 2e9)\
                if np.random.rand() < tol_t_burst else\
                (2e9, self.t_form)
        else:
            domain_t_burst = (self.domain_t_burst[0]+self.t_ext, self.t_form)

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

    def draw_redshift(self, get_max_redshift=True, max_survey=3.0):
        if self.t_form is None:
            self.draw_t_form()

        maximum_redshift = cosmology.z_at_value(
            UNIVERSE.age, (UNIVERSE_AGE-self.t_form)*u.yr
        )
        if get_max_redshift:
            if max_survey is None or max_survey > maximum_redshift:
                self.redshift = maximum_redshift
            else:
                self.redshift = max_survey
        else:
            if max_survey is not None:
                maximum_redshift = min(maximum_redshift, max_survey)
            domain_redshift = (self.domain_redshift[0], maximum_redshift)
            self.redshift = _random_range_(domain_redshift)
        return self.redshift

    def get_samples(self, size=1, pristine=False):
        self._clean_draws_()

        columns = ["t_form", "gamma", "truncated", "t_trun", "tau_trun",
                   "t_burst", "t_ext", "a_burst", "metallicity", "tau_v",
                   "mu_v", "sigma_v", "redshift"]
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
            sample["redshift"] += [self.draw_redshift()]
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
        # read all filters
        self.filter_responses = pd.read_csv("../data/filters.csv")
        # read all models in default path
        self.models.set_all()

        # initial value for library
        self.sfhs = None
        # initial value for SEDs
        self.seds_nebular = None
        self.seds_stellar = None
        # initial value for physical
        self.physical = None

    def get_extinction_curve(self, iloc, timescale):
        """Build extinction curve from Charlot & Fall (2001)."""
        wl = self.models.wavelength
        tau_v = self.sample.tau_v[iloc]
        mu_v = self.sample.mu_v[iloc]
        n_bc = np.count_nonzero(timescale < 1e7)
        n_ism = timescale.size - n_bc

        ext_bc = np.tile(np.exp(-tau_v*(wl/5500.0)**(-0.7)),
                         (n_bc, 1)).T
        ext_ism = np.tile(np.exp(-mu_v*tau_v*(wl/5500.0)**(-0.7)),
                          (n_ism, 1)).T

        return np.column_stack((ext_bc, ext_ism))

    def get_kinematic_effects(self, iloc, sed):
        c_mks = 3e5
        m = 6.0
        nx = 100
        wl = self.models.wavelength
        nwl = wl.size + 2*nx
        losvd = self.sample.sigma_v[iloc]

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
            m2 = np.min([j+1, nwl-1])
            m1 = np.max([0, 2*i-m2])

            u = (wl_[i] / wl_[m2:m1:-1] - 1.0) * c_mks
            g = norm.pdf(u, loc=0.0, scale=losvd)
            w = np.trapz(g, u)
            g = sed_[m2:m1:-1] * g / (w if w > 0.0 else 1.0)

            if i >= nx and i < nwl-nx:
                sed_[i] = np.trapz(g, x=u)

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
                # find column k such that:
                # t[k-1] < t_new[i] < t[k]
                k = np.searchsorted(t, t_new[i])
                # interpolate in models (k-1, k) assuming linearity in log-ages

                t_0, t_1, t_2 = np.log10([t_new[i], t[k-1], t[k]])
                v, w = (t_2 - t_0)/(t_2 - t_1), (t_0 - t_1)/(t_2 - t_1)
                for j in xrange(len(ssps)):
                    ssps[j].insert(k, t_new[i],
                                   v*ssps[j].get(t[k-1]) +
                                   w*ssps[j].get(t[k]))

        # return models dropping the older than galaxy ones
        return [ssps[j].drop(columns=t[t > t_new[0]])
                for j in xrange(len(ssps))]

    def get_physical_properties(self, iloc, sfh, ssps, passband=("SDSS", "r")):
        timescale = ssps[0].columns.values
        delta_t = np.diff(np.concatenate(([0.0], timescale)))

        sdss_r_id = self.filter_responses.groupby(
            ("survey", "filter")
        ).groups.get(passband).values
        sdss_r = self.filter_responses.iloc[sdss_r_id].get(
            ["wavelength", "response"]).values
        lum = np.array([iflux(np.column_stack(
            (ssps[1].index, ssps[1].iloc[:, j])), sdss_r)
            for j in xrange(ssps[1].columns.size)]
        )

        mass_bins = sfh * delta_t

        physical = OrderedDict()
        physical["log_stellar_mass"] = np.log10(mass_bins.sum())
        physical["log_ssfr_10myr"] = np.log10(mass_bins[timescale <= 1e7].sum()
                                              / 1e7 / mass_bins.sum())
        physical["logt_l"] = np.average(np.log10(timescale),
                                        weights=lum*mass_bins)
        physical["logt_m"] = np.average(np.log10(timescale), weights=mass_bins)
        physical["logz_l"] = np.log10(self.sample.metallicity[iloc])
        physical["logz_m"] = np.log10(self.sample.metallicity[iloc])
        physical["av_eff"] = 1.086*self.sample.tau_v[iloc] *\
            self.sample.mu_v[iloc]
        physical["sigma_v"] = self.sample.sigma_v[iloc]

        physical = pd.DataFrame(physical, index=(iloc,))

        return physical

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

        mask_cont = np.ones(timescale.size, dtype=np.bool)
        # mask ages older than galaxy
        mask_cont[timescale > t_form] = False
        SFH_trun = np.zeros(timescale.size)
        if self.sample.truncated[iloc]:
            # mask ages younger than truncation time
            mask_cont[timescale <= t_trun] = False
            # mask ages older than truncation time in truncated SFH
            mask_trun = np.ones(timescale.size, dtype=np.bool)
            mask_trun[timescale > t_trun] = False
            # build truncation (SFH_trun)
            SFH_trun[mask_trun] = np.exp(-(t_trun - timescale[mask_trun]) /
                                         tau_trun)
            SFH_trun[mask_trun] *= np.exp(-(t_form - t_trun) * gamma*1e-9)
        SFH_cont = np.zeros(timescale.size)
        # build main SFH (SFH_cont)
        SFH_cont[mask_cont] = np.exp(-(t_form - timescale[mask_cont]) *
                                     gamma*1e-9)
        # compute mass without burst
        mass_under = np.trapz(SFH_cont+SFH_trun, x=timescale)
        # maskout burst times from underlying SFH
        mask_cont[(t_burst_f <= timescale) & (timescale <= t_burst_i)] = False
        SFH_cont[~mask_cont] = 0.0
        mask_burst = np.ones(timescale.size, dtype=np.bool)
        # mask ages older/younger than burst interval
        mask_burst[timescale < t_burst_f] = False
        mask_burst[timescale > t_burst_i] = False
        SFH_burst = np.zeros(timescale.size)
        # build burst (SFH_burst)
        SFH_burst[mask_burst] = mass_under * a_burst / (t_burst_i - t_burst_f)

        # build full SFH
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
        """Build both: the SFHs and the SEDs."""
        sfhs = OrderedDict()
        seds = []
        columns = self.sample.index
        wl = self.models.wavelength
        for i in columns:
            ssps = self.get_metallicity_interpolation(i, emission=emission)
            ssps = self.get_time_interpolation(i, ssps)
            sfhs[i] = self.get_sfh(i, ssps[0].columns)
            timescale = sfhs[i].index.values
            delta_t = np.diff(np.concatenate(([0.0], timescale)))
            for j in xrange(len(ssps)):
                ssps[j] *= self.get_extinction_curve(i, timescale)
                mass_bins = sfhs[i] * delta_t
                seds += [np.average(ssps[j].values,
                                    weights=np.tile(mass_bins, (wl.size, 1)),
                                    axis=1)]
                seds[-1] = self.get_kinematic_effects(i, seds[-1])
            if self.physical is None:
                self.physical = self.get_physical_properties(i, sfhs[i], ssps)
            else:
                self.physical = self.physical.append(
                    self.get_physical_properties(i, sfhs[i], ssps),
                    ignore_index=True
                )
        self.sfhs = sfhs
        if emission == "both":
            self.seds_nebular = pd.DataFrame(np.array(seds[::2]).T,
                                             index=wl,
                                             columns=columns)
            self.seds_stellar = pd.DataFrame(np.array(seds[1::2]).T,
                                             index=wl,
                                             columns=columns)
        elif emission == "nebular":
            self.seds_nebular = pd.DataFrame(np.array(seds).T,
                                             index=wl,
                                             columns=columns)
        elif emission == "stellar":
            self.seds_stellar = pd.DataFrame(np.array(seds).T,
                                             index=wl,
                                             columns=columns)
        return None
