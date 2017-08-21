import numpy as np

def cmb_CE(ell, Cl):
    mf = modal_fraction(ell,Cl)
    mf[mf<0] = np.nan
    ce, ced = ce_density(mf)
    return {'mf':mf,'ced':ced,'ce':ce}

def modal_fraction(ell,Cl):
    nCl = (2*ell+1)*Cl
    return nCl/np.nansum(nCl)

def ce_density(mf):
    ced = -mf*np.log2(mf)
    ce = np.nansum(ced)
    return ce, ced

