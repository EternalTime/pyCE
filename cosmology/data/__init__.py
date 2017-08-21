#------------------------------------------------------------------ LIBRARIES

import urllib2 as ul
import numpy as np
from astropy.io import fits

#------------------------------------------------------------------- FUNCTIONS

def get_data_power_spectrum(telescope):
    try:
        if telescope in ['Planck','planck']:
            url = ('http://irsa.ipac.caltech.edu/data/Planck/release_2/anci' +
                  'llary-data/cosmoparams/COM_PowerSpect_CMB_R2.02.fits')
            hdulist = fits.open(url)
            ell = (np.array(map(float,np.append(hdulist['TTLOLUNB'].
                data.field(0),hdulist['TTHILUNB'].data.field(0)))))
            Dl  = np.append(hdulist['TTLOLUNB'].data.field(1),
                hdulist['TTHILUNB'].data.field(1))
        elif telescope in ['WMAP','wmap']:
            url = ('https://lambda.gsfc.nasa.gov/data/map/dr5/dcp/spectra/' +
                  'wmap_tt_spectrum_9yr_v5.txt')
            data = ul.urlopen(url) #open the external data txt file
            data = [line.split() for line in data] #break apart each line
            data = np.transpose([map(float,line) for line in data[20::]])
            ell,Dl = data[0],data[1]
        else:
            print('No best fit angular power spectrum for that telescope')

        return {'ell':ell,'C':2*np.pi*Dl/(ell*(ell+1))}
    except:
        print('error in pyCE.access_data.online.get_data_power_spectrum')
