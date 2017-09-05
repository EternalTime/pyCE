#!/usr/bin/python
# Filename: matter_power_spectrum_test.py

if __name__ == '__main__':

    import sys, platform, os, camb
    from matplotlib import pyplot as plt
    import numpy as np
    from camb import model, initialpower

    #Now get matter power spectra and sigma8 at redshift 0 and 0.8
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
    pars.set_dark_energy() #re-set defaults
    pars.InitPower.set_params(ns=0.965)
    #Not non-linear corrections couples to smaller scales than you want
    pars.set_matter_power(redshifts=list(np.linspace(0,1200,1)), kmax=100.0)

    #Linear spectra
    pars.NonLinear = model.NonLinear_none
    results = camb.get_results(pars)
    kh, z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=100, npoints = 5000)
    s8 = np.array(results.get_sigma8())

    #Non-Linear spectra (Halofit)
    #pars.NonLinear = model.NonLinear_both
    #results.calc_power_spectra(pars)
    #kh_nonlin, z_nonlin, pk_nonlin = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1, npoints = 200)

    for i, (redshift) in enumerate(z):
        plt.loglog(kh, pk[i,:], color='k', ls = '-')
        #plt.loglog(kh_nonlin, pk_nonlin[i,:], color='r', ls = '-')
    plt.xlabel('k/h Mpc');
    plt.title('Matter power spectrum');
    plt.show()

    if len(pk)==1:
        pk  = np.reshape(pk,np.size(pk))
