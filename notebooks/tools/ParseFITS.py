import numpy as np
import pandas as pd

def merge_stellar_nebular(fits_object):
    #wl_ste = array([fits_object[0].header["CRVAL1"]+i*fits_object[0].header["CDELT1"] for i in xrange(fits_object[0].header["NAXIS1"])])
    wl_ste = fits_object[3].data["BFIT"]
    fl_ste = fits_object[0].data.T
    #wl_ste = gaussian_filter(wl, 0.4247*fits_object[0].header["H_WRESOL"])
    wl_neb = fits_object[1].data["WAVE"]
    fl_neb = fits_object[1].data["FLUXLINE"]

    wl_tot = np.concatenate((wl_ste,wl_neb))
    wl_sor, uniq_wl = np.unique(wl_tot, return_index=True)
    # interpolate stellar SED in lines wavelenths
    # fill with zeros SED fluxes in line wavelengths beyond original SED
    fl_ste_tot = np.zeros((uniq_wl.size, fl_ste.shape[1]))
    for j in xrange(fl_ste_tot.shape[1]): fl_ste_tot[:,j] = np.interp(wl_sor, wl_ste, fl_ste[:,j], left=0.0, right=0.0)
    # fill with zeros line fluxes in SED wavelengths
    fl_neb_tot = np.zeros((wl_tot.size, fl_ste.shape[1]))
    fl_neb_tot[-wl_neb.size:] = fl_neb
    fl_neb_tot = fl_neb_tot[uniq_wl]

    ages = pd.Series(fits_object[2].data["AGE"]*10**6)
    ts_nms = ["%.0f Myr"%(ages[i]/10**6) for i in xrange(fl_neb_tot.shape[1])]
    SEDs_ste = pd.DataFrame(fl_ste_tot, index=wl_sor, columns=ts_nms)
    SEDs_tot = pd.DataFrame(fl_ste_tot+fl_neb_tot, index=wl_sor, columns=ts_nms)

    return ages, SEDs_ste, SEDs_tot
