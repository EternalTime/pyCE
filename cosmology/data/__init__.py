#--------------------------------------------------------------------- LIBRARIES

import urllib2 as ul
import numpy as np
import os
from astropy.io import fits

#--------------------------------------------------------- INITIALIZATION SCRIPT

directory = os.path.dirname(__file__)

#--------------------------------------------------------------------- FUNCTIONS

def read_power_spectrum(telescope = 'Planck', psType = 'TT'):
    try:
        if telescope in ['Planck','planck']:
            url = ('http://irsa.ipac.caltech.edu/data/Planck/release_2/anci' +
                  'llary-data/cosmoparams/COM_PowerSpect_CMB_R2.02.fits')
            hdulist = fits.open(url)
            if psType in ['TT','tt']:
                ell = (np.array(map(float,np.append(hdulist['TTLOLUNB'].
                    data.field(0),hdulist['TTHILUNB'].data.field(0)))))
                Dl  = np.append(hdulist['TTLOLUNB'].data.field(1),
                    hdulist['TTHILUNB'].data.field(1))
        elif telescope in ['WMAP','wmap']:
            if psType in ['TT','tt']:
                fileName = os.path.join(directory,
                           'wmap/wmap_tt_spectrum_9yr_v5.txt')
                print(fileName)
                myFile = open(fileName)
                data = [line for line in myFile.readlines()]
                data = [line.split() for line in data]
                data = np.transpose([map(float,line) for line in data[20::]])
                ell,Dl = data[0],data[1]
        return {'ell':ell,'Dl':Dl,'Cl':2*np.pi*Dl/(ell*(ell+1))}
    except:
        print('Something went wrong reading the file')
