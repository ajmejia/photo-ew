import numpy as np

def integrated_flux(SED, passband):
    mask = (passband[0,0] <= SED[:,0])&(SED[:,0] <= passband[-1,0])
    ipassband = np.interp(SED[mask,0], passband[:,0], passband[:,1])

    return np.trapz(SED[mask,1]*ipassband, SED[mask,0])/np.trapz(ipassband, SED[mask,0])

def ABmag(SED, passband):
    SED_AB = np.zeros(SED.shape)
    SED_AB[:,0] = SED[:,0]
    SED_AB[:,1] = (3.6308e-20/SED_AB[:,0]**2)*(3e18/3.828e33)

    F_lambda = integrated_flux(SED, passband)
    C_lambda = integrated_flux(SED_AB, passband)
    return -2.5*np.log10(F_lambda/C_lambda)
