#--------------------------------------------------------------------- LIBRARIES

import urllib2 as ul
import numpy as np
import os
from astropy.io import fits

#--------------------------------------------------------- INITIALIZATION SCRIPT

directory = os.path.dirname(__file__)

#--------------------------------------------------------------------- FUNCTIONS

def read_power_spectrum(telescope = 'Planck', ps = 'TT', psType = 'data'):
    """
    ----------------------------------------------------------------------------
    FUNCTION:       dic = read_power_spectrum([telescope, ps, psType])
    ----------------------------------------------------------------------------
    INPUT:          telescope (string: Planck or WMAP)
                    ps        (string: TT, TE, TM, or EM)
                    psType    (string: data, fit)
    ----------------------------------------------------------------------------
    OUTPUT:         dic (dictionary, keys: ell, Dl, Cl)
                    Outputs a dictionary containing the ell, D_ell, and C_ell
                    values coming directly from the data from either Planck
                    or WMAP.
    ----------------------------------------------------------------------------
    """
    try:
        if psType in ['data','Data','D']:
            if telescope in ['P','p','Planck','planck']:
                url = ('http://irsa.ipac.caltech.edu/data/Planck/release_2/anci' +
                      'llary-data/cosmoparams/COM_PowerSpect_CMB_R2.02.fits')
                hdulist = fits.open(url)
                if ps in ['TT','tt']:
                    ell = (np.array(map(float,np.append(hdulist['TTLOLUNB'].
                        data.field(0),hdulist['TTHILUNB'].data.field(0)))))
                    Dl  = np.append(hdulist['TTLOLUNB'].data.field(1),
                        hdulist['TTHILUNB'].data.field(1))
                    error  = np.append(hdulist['TTLOLUNB'].data.field(2),
                        hdulist['TTHILUNB'].data.field(2))
            elif telescope in ['WMAP','wmap']:
                if ps in ['TT','tt']:
                    fileName = os.path.join(directory,
                               'wmap/wmap_tt_spectrum_9yr_v5.txt')
                    print(fileName)
                    myFile = open(fileName)
                    data = [line for line in myFile.readlines()]
                    data = [line.split() for line in data]
                    data = np.transpose([map(float,line) for line in data[20::]])
                    ell,Dl,error = data[0],data[1],data[2]
            return {'ell':ell,'Dl':Dl,'Cl':2*np.pi*Dl/(ell*(ell+1)),'error':error}
        elif psType in ['bf','best','bestfit','fit']:
            if telescope in ['P','p','Planck','planck']:
                url = ('http://irsa.ipac.caltech.edu/data/Planck/release_2/' +
                      'ancillary-data/cosmoparams/COM_PowerSpect_CMB-base-' +
                      'plikHM-TT-lowTEB-minimum-theory_R2.02.txt')
                data = ul.urlopen(url).readlines()[1::]
                data = np.array(map(lambda x:map(float,x.split()),data))
                ell = data[:,0]
                if ps in ['TT','tt']:
                    Dl = data[:,1]
            return {'ell':ell,'Dl':Dl,'Cl':2*np.pi*Dl/(ell*(ell+1))}
    except:
        print('Something went wrong reading the file')
