import numpy as np

def equivalent_width(SED, centroids, widths):
    wl, fl = SED.T

    wl_in = np.array(centroids)-np.array(widths)*0.5
    wl_fi = np.array(centroids)+np.array(widths)*0.5

    EWs = np.zeros(len(centroids))
    for j in xrange(EWs.size):
        if wl_in[j]<wl[0] or wl_fi[j]>wl[-1]:
            EWs[j] = np.nan
            continue

        mask = (wl_in[j]<=wl)&(wl<=wl_fi[j])
        wl_m, fl_m = wl[mask], fl[mask]

        fl_c = fl_m[0] + (fl_m[-1]-fl_m[0])/(wl_m[-1]-wl_m[0])*(wl_m-wl_m[0])

        EWs[j] = np.trapz(1-fl_m/fl_c, wl_m)

    return np.abs(EWs)
