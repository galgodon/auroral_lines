# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 11:41:24 2019

@author: Gerome Algodon

Creating table of data for Winter 2019 Research project - Finding Auroral Lines
"""

import numpy as np
from astropy.io import fits
from astropy.table import Table

import sys

import warnings                    # I put this here so the invalid value in sqrt warning would be be ignored
warnings.filterwarnings("ignore")

# =============================================================================
# Start with an array of Plate-Ifu's to make the table for
# =============================================================================

# Though we will change this later, I will for now use the sample of galaxies 
# Francesco gave me over the summer, We just need a new list of plate and ifu in order to change the table

log = fits.open('data/MPL-6_master_catalogue_Apr9.fits')

eLIER = log[1].data['BPT_C'] == 3

indx = eLIER&np.invert((log[1].data['PLATE-IFU']=='8146-3702')|(log[1].data['PLATE-IFU']=='8158-3703'))  
    # these two galaxies are not in MPL-7 I figured this out using check_galaxy.py from my summer code
plate = log[1].data['PLATE'][indx]
ifu = log[1].data['IFUDESIGN'][indx]

# =============================================================================
# Initialize Empty Table
# =============================================================================

dtype = [('Plate', 'i4'), ('IFU', 'i4'), ('stell_vel_ID', 'f8'),('emline_ID','f8'),('rad_norm_spx','f8'),
         ('azimuth_spx','f8'),('rad_norm_bin','f8'),('azimuth_bin','f8'),('snr_spx','f8'),
         ('snr_bin','f8'),('stell_vel','f8'),('stell_vel_mask','f8'),('stell_sigma_cor','f8'),
         ('stell_sigma_mask','f8'),('spec_index','f8'),('spec_index_err','f8'),('spec_index_mask','f8')]
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

# =============================================================================
# Now we need to pull all of the data we need for each spaxel in each galaxy
# =============================================================================

for i in range(len(plate)):
    
    print('Galaxy {}/{}'.format(i+1,len(plate)))
    maps_file = '/data/manga/spectro/analysis/MPL-7/HYB10-GAU-MILESHC/{0}/{1}/manga-{0}-{1}-MAPS-HYB10-GAU-MILESHC.fits.gz'.format(plate[i],ifu[i])
    maps = fits.open(maps_file)
    
    tot_spaxel = len(maps['BINID'].data[1,:,:].flatten())   # total number of spaxels in this galaxy
    
    for l in range(tot_spaxel):    # loop through each spaxel

        sys.stdout.write('\r'+('   Importing data from spaxel {}/{}'.format(l+1,tot_spaxel)))
        row = []                   # initialize row, we will append the data for each column
        row.append(plate[i])       # note plate and ifu are indexed by 'i', this makes it so every spaxel in the 
        row.append(ifu[i])            # same galaxy has the same plate and ifu
    
        # Get Bin IDs we want
        row.append(maps['BINID'].data[1,:,:].flatten()[l])   # stellar velocity ID
        row.append(maps['BINID'].data[3,:,:].flatten()[l])   # emline and d4000 ID
    
        # Get coordinates
        row.append(maps['SPX_ELLCOO'].data[1,:,:].flatten()[l])  # radius normalized by the elliptical Petrosian effective radius from the NSA
        row.append(maps['SPX_ELLCOO'].data[2,:,:].flatten()[l])  # azimuth angle
        
        row.append(maps['BIN_LWELLCOO'].data[1,:,:].flatten()[l])
        row.append(maps['BIN_LWELLCOO'].data[2,:,:].flatten()[l])
        
        # Get continuum signal to noise ratio
        row.append(maps['SPX_SNR'].data.flatten()[l])  # S/N in each spaxel (emission-line measurements done per spaxel)
        row.append(maps['BIN_SNR'].data.flatten()[l])  # S/N in each bin (stellar measurements done per bin)
        
        # Get stellar velocity
        row.append(maps['STELLAR_VEL'].data.flatten()[l])
        row.append(maps['STELLAR_VEL_MASK'].data.flatten()[l])
        
        # Get corrected velocity dispersion
        row.append((np.sqrt( (maps['STELLAR_SIGMA'].data)**2 - (maps['STELLAR_SIGMACORR'].data)**2 )).flatten()[l])
        row.append(maps['STELLAR_SIGMA_MASK'].data.flatten()[l])
        
        # Get spectral index for D4000 (indx 43, channel 44)
        row.append(maps['SPECINDEX'].data[43,:,:].flatten()[l])
        row.append(np.power(maps['SPECINDEX_IVAR'].data[43,:,:],-1/2).flatten()[l])
        row.append(maps['SPECINDEX_MASK'].data[43,:,:].flatten()[l])
    
        # Get emmision line properties of Ha, hb, o2, n2, s2, o3, o1
        
        for j in range(len(emline_names)):
            row.append(maps['EMLINE_SEW'].data[emline_indx[j],:,:].flatten()[l])                        # summed equivalent width
            row.append(maps['EMLINE_SEW_MASK'].data[emline_indx[j],:,:].flatten()[l])            
            row.append(maps['EMLINE_GFLUX'].data[emline_indx[j],:,:].flatten()[l])                      # gaussian flux
            row.append(np.power(maps['EMLINE_GFLUX_IVAR'].data[emline_indx[j],:,:],-1/2).flatten()[l])  # gaussian flux error
            row.append(maps['EMLINE_GFLUX_MASK'].data[emline_indx[j],:,:].flatten()[l])                 # same mask for flux and error
        
        table.add_row(row)
    
table.write('data/spaxel_data_table.fits', format='fits')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
