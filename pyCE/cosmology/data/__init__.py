"""Access to CMB angular power spectrum data.

Loads observed angular power spectra either from the bundled WMAP 9-year
files (in the ``wmap/`` subdirectory of this package) or by downloading the
Planck release-2 products from IRSA over the network.
"""

#--------------------------------------------------------------------- LIBRARIES

from urllib import request as ul
import numpy as np
import os
from astropy.io import fits

#--------------------------------------------------------- INITIALIZATION SCRIPT

#absolute path of this package; used to locate the bundled WMAP files
directory = os.path.dirname(__file__)

#--------------------------------------------------------------------- FUNCTIONS

def read_power_spectrum(telescope = 'Planck', ps = 'TT', psType = 'data'):
    """Read a CMB angular power spectrum.

    Parameters
    ----------
    telescope : str
        'Planck' (downloaded from IRSA; requires network access) or
        'WMAP' (9-year data bundled with the package).
    ps : str
        Which spectrum to read. Currently only 'TT' is implemented.
    psType : str
        'data' for the observed spectrum, or 'fit'/'bestfit' for the
        Planck best-fit theory spectrum (Planck only).

    Returns
    -------
    dict
        Keys 'ell', 'Dl', 'Cl' (with Cl = 2*pi*Dl/(ell*(ell+1))), and, for
        psType='data', 'error'. Prints a message and returns None if the
        read fails.
    """
    try:
        if psType in ['data','Data','D']:
            if telescope in ['P','p','Planck','planck']:
                url = ('http://irsa.ipac.caltech.edu/data/Planck/release_2/anci' +
                      'llary-data/cosmoparams/COM_PowerSpect_CMB_R2.02.fits')
                hdulist = fits.open(url)
                if ps in ['TT','tt']:
                    #low-ell and high-ell unbinned spectra, concatenated
                    ell = (np.append(hdulist['TTLOLUNB'].data.field(0),
                        hdulist['TTHILUNB'].data.field(0)).astype(float))
                    Dl  = np.append(hdulist['TTLOLUNB'].data.field(1),
                        hdulist['TTHILUNB'].data.field(1))
                    error  = np.append(hdulist['TTLOLUNB'].data.field(2),
                        hdulist['TTHILUNB'].data.field(2))
            elif telescope in ['WMAP','wmap']:
                if ps in ['TT','tt']:
                    fileName = os.path.join(directory,
                               'wmap/wmap_tt_spectrum_9yr_v5.txt')
                    myFile = open(fileName)
                    data = [line for line in myFile.readlines()]
                    data = [line.split() for line in data]
                    #skip the 20-line header
                    data = np.transpose([[float(v) for v in line] for line in data[20::]])
                    ell,Dl,error = data[0],data[1],data[2]
            return {'ell':ell,'Dl':Dl,'Cl':2*np.pi*Dl/(ell*(ell+1)),'error':error}
        elif psType in ['bf','best','bestfit','fit']:
            if telescope in ['P','p','Planck','planck']:
                url = ('http://irsa.ipac.caltech.edu/data/Planck/release_2/' +
                      'ancillary-data/cosmoparams/COM_PowerSpect_CMB-base-' +
                      'plikHM-TT-lowTEB-minimum-theory_R2.02.txt')
                data = [line.decode() for line in ul.urlopen(url).readlines()[1::]]
                data = np.array([[float(v) for v in line.split()] for line in data])
                ell = data[:,0]
                if ps in ['TT','tt']:
                    Dl = data[:,1]
            return {'ell':ell,'Dl':Dl,'Cl':2*np.pi*Dl/(ell*(ell+1))}
    except:
        print('Something went wrong reading the file')
