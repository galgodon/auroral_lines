# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 14:40:34 2019

@author: gerom
"""
import numpy as np
from astropy.io import fits
from astropy.table import Table

import sys

import warnings
warnings.filterwarnings("ignore")

plate = np.array([7977,8143])
ifu = np.array([12705,3701])
dtype = [('plate', 'i4'), ('ifu', 'i4'),('z_vel', 'f8'),('gal_red_B-V', 'f8'), ('stell_vel_ID', 'f8'),('emline_ID','f8'),('rad_norm_spx','f8'),
         ('azimuth_spx','f8'),('snr_spx','f8'),
         ('snr_bin','f8'),('stell_vel','f8'),('stell_vel_mask','f8'),('ha_vel', 'f8'),('ha_vel_mask', 'f8'),('stell_sigma_cor','f8'),
         ('stell_sigma_mask','f8'),('spec_index_D4000','f8'),('spec_index_ivar_D4000','f8'),('spec_index_mask_D4000','f8'),('spec_index_HDeltaA', 'f8'),('spec_index_ivar_HDeltaA', 'f8'),('spec_index_mask_HDeltaA', 'f8')]
emline_names = np.array(['H_alpha','H_beta','OII-3727','OII-3729','NII-6549','SII-6718','SII-6732','OIII-4960','OIII-5008','OI-6302','OI-6365'])
emline_indx = np.array([18,11,0,1,17,20,21,12,13,15,16])  # corresponding index numbers found on https://trac.sdss.org/wiki/MANGA/TRM/TRM_MPL-7/DAPDataModel#DAPMAPSfile
assert len(emline_names)==len(emline_indx)
for k in range(len(emline_names)):
    dtype.append(('summed_EW_{}'.format(emline_names[k]),'f8'))
    dtype.append(('summed_EW_mask_{}'.format(emline_names[k]),'f8'))
    dtype.append(('gaus_flux_{}'.format(emline_names[k]),'f8'))
    dtype.append(('gaus_ferr_{}'.format(emline_names[k]),'f8'))
    dtype.append(('gaus_flux_mask_{}'.format(emline_names[k]),'f8'))
table = Table(dtype=dtype)

for i in range(len(plate)):
    
    print('Galaxy {}/{}'.format(i+1,len(plate)))
    
    maps_file = 'data/manga-{0}-{1}-MAPS-HYB10-GAU-MILESHC.fits.gz'.format(plate[i],ifu[i])
    maps = fits.open(maps_file)
    
    good_spaxel = (maps['BINID'].data[0,:,:].flatten() != -1)   # 

    row = np.zeros((good_spaxel.sum(),len(dtype)))
    
    row[:,0] = (np.full(good_spaxel.sum(),plate[i]))       
    row[:,1] = (np.full(good_spaxel.sum(),ifu[i]))
    
    row[:,2] = (np.full(good_spaxel.sum(),maps[0].header['SCINPVEL']))
    row[:,3] = (np.full(good_spaxel.sum(),maps[0].header['EBVGAL']))

    # Get Bin IDs we want
    row[:,4] = (maps['BINID'].data[1,:,:].flatten()[good_spaxel])
    row[:,5] = (maps['BINID'].data[3,:,:].flatten()[good_spaxel])

    # Get coordinates
    row[:,6] = (maps['SPX_ELLCOO'].data[1,:,:].flatten()[good_spaxel])  # radius normalized by the elliptical Petrosian effective radius from the NSA
    row[:,7] = (maps['SPX_ELLCOO'].data[2,:,:].flatten()[good_spaxel])  # azimuth angle
    
    # Get continuum signal to noise ratio
    row[:,8] = (maps['SPX_SNR'].data.flatten()[good_spaxel])  # S/N in each spaxel (emission-line measurements done per spaxel)
    row[:,9] = (maps['BIN_SNR'].data.flatten()[good_spaxel])  # S/N in each bin (stellar measurements done per bin)
    
    # Get stellar velocity
    row[:,10] = (maps['STELLAR_VEL'].data.flatten()[good_spaxel])
    row[:,11] = (maps['STELLAR_VEL_MASK'].data.flatten()[good_spaxel])
    
    # Get H_alpha velocity
    row[:,12] = (maps['EMLINE_GVEL'].data[18,:,:].flatten()[good_spaxel])
    row[:,13] = (maps['EMLINE_GVEL_MASK'].data[18,:,:].flatten()[good_spaxel])

    # Get corrected velocity dispersion
    row[:,14] = ((np.sqrt( (maps['STELLAR_SIGMA'].data)**2 - (maps['STELLAR_SIGMACORR'].data)**2 )).flatten()[good_spaxel])
    row[:,15] = (maps['STELLAR_SIGMA_MASK'].data.flatten()[good_spaxel])
    
    # Get spectral index for D4000 (indx 43, channel 44)
    row[:,16] = (maps['SPECINDEX'].data[43,:,:].flatten()[good_spaxel])
    row[:,17] = (maps['SPECINDEX_IVAR'].data[43,:,:].flatten()[good_spaxel])
    row[:,18] = (maps['SPECINDEX_MASK'].data[43,:,:].flatten()[good_spaxel])

    # Get spectral index for HDeltaA (indx 21, channel 22)
    row[:,19] = (maps['SPECINDEX'].data[21,:,:].flatten()*maps['SPECINDEX_CORR'].data[21,:,:].flatten())[good_spaxel]
    row[:,20] = (maps['SPECINDEX_IVAR'].data[21,:,:].flatten()[good_spaxel])
    row[:,21] = (maps['SPECINDEX_MASK'].data[21,:,:].flatten()[good_spaxel])
    
    # Get emmision line properties of Ha, hb, o2, n2, s2, o3, o1
    
    for j in range(len(emline_names)):
        row[:,22+5*j] = (maps['EMLINE_SEW'].data[emline_indx[j],:,:].flatten()[good_spaxel])          # summed equivalent width
        row[:,23+5*j] = (maps['EMLINE_SEW_MASK'].data[emline_indx[j],:,:].flatten()[good_spaxel])            
        row[:,24+5*j] = (maps['EMLINE_GFLUX'].data[emline_indx[j],:,:].flatten()[good_spaxel])        # gaussian flux
        row[:,25+5*j] = (maps['EMLINE_GFLUX_IVAR'].data[emline_indx[j],:,:].flatten()[good_spaxel])   # gaussian flux error
        row[:,26+5*j] = (maps['EMLINE_GFLUX_MASK'].data[emline_indx[j],:,:].flatten()[good_spaxel])   # same mask for flux and error
        
    for l in range(len(row)):
        sys.stdout.write('\r'+('  adding row {}/{}'.format(l+1,len(row))))
        table.add_row(row[l])
    print()

table.write('test2.fits', format='fits')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    