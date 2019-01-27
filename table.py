# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 11:41:24 2019

@author: Gerome Algodon

Creating table of data for Winter 2019 Research project - Finding Auroral Lines
"""

import numpy as np
from astropy.io import fits
from astropy.table import Table

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
dtype = [('Plate', 'i4'), ('IFU', 'i4'), ('stell_vel_ID', 'object'),('emline_ID','object'),('rad_norm_spx','object'),
         ('azimuth_spx','object'),('rad_norm_bin','object'),('azimuth_bin','object'),('snr_spx','object'),
         ('snr_bin','object'),('stell_vel','object'),('stell_vel_mask','object'),('stell_sigma_cor','object'),
         ('stell_sigma_mask','object'),('spec_index','object'),('spec_index_err','object'),('spec_index_mask','object')]
emline_names = np.array(['H_alpha','H_beta','OII-3727','OII-3729','NII-6549','SII-6718','SII-6732','OIII-4960','OIII-5008','OI-6302','OI-6365'])
emline_indx = np.array([18,11,0,1,17,20,21,12,13,15,16])  # corresponding index numbers found on https://trac.sdss.org/wiki/MANGA/TRM/TRM_MPL-7/DAPDataModel#DAPMAPSfile
assert len(emline_names)==len(emline_indx)
for k in range(len(emline_names)):
    dtype.append(('summed_EW_{}'.format(emline_names[k]),'object'))
    dtype.append(('summed_EW_mask_{}'.format(emline_names[k]),'object'))
    dtype.append(('gaus_flux_{}'.format(emline_names[k]),'object'))
    dtype.append(('gaus_ferr_{}'.format(emline_names[k]),'object'))
    dtype.append(('gaus_flux_mask_{}'.format(emline_names[k]),'object'))
table = Table(data=np.zeros(len(plate),dtype))
# =============================================================================
# Now we need to pull all of the data we need for each galaxy
# =============================================================================

for i in range(len(plate)):
    
    table['Plate'][i] = plate[i]
    table['IFU'][i] = ifu[i]
    
    maps_file = '/data/manga/spectro/analysis/MPL-7/HYB10-GAU-MILESHC/{0}/{1}/manga-{0}-{1}-MAPS-HYB10-GAU-MILESHC.fits.gz'.format(plate[i],ifu[i])
    maps = fits.open(maps_file)
    
    # Get Bin IDs we want
    table['stell_vel_ID'][i] = maps['BINID'].data[1,:,:]
    table['emline_ID'][i] = maps['BINID'].data[3,:,:]
    
    # Get coordinates
    table['rad_norm_spx'][i] = maps['SPX_ELLCOO'].data[1,:,:]  # radius normalized by the elliptical Petrosian effective radius from the NSA
    table['azimuth_spx'][i] =  maps['SPX_ELLCOO'].data[2,:,:]  # azimuth angle
    
    table['rad_norm_bin'][i] = maps['BIN_LWELLCOO'].data[1,:,:]
    table['azimuth_bin'][i] =  maps['BIN_LWELLCOO'].data[2,:,:]
    
    # Get continuum signal to noise ratio
    table['snr_spx'][i] = maps['SPX_SNR'].data  # S/N in each spaxel (emission-line measurements done per spaxel)
    table['snr_bin'][i] = maps['BIN_SNR'].data  # S/N in each bin (stellar measurements done per bin)
    
    # Get stellar velocity
    table['stell_vel'][i] = maps['STELLAR_VEL'].data
    table['stell_vel_mask'][i] = maps['STELLAR_VEL_MASK'].data
    
    # Get corrected velocity dispersion
    table['stell_sigma_cor'][i] = np.sqrt( (maps['STELLAR_SIGMA'].data)**2 - (maps['STELLAR_SIGMACORR'].data)**2 )
    table['stell_sigma_mask'][i] = maps['STELLAR_SIGMA_MASK'].data
    
    # Get spectral index for D4000 (indx 43, channel 44)
    table['spec_index'][i] = maps['SPECINDEX'].data[43,:,:]
    table['spec_index_err'][i] = np.power(maps['SPECINDEX_IVAR'].data[43,:,:],-1/2)
    table['spec_index_mask'][i] = maps['SPECINDEX_MASK'].data[43,:,:]
    
    # Get emmision line properties of Ha, hb, o2, n2, s2, o3, o1
    
    for j in range(len(emline_names)):
        table['summed_EW_{}'.format(emline_names[j])][i] = maps['EMLINE_SEW'].data[emline_indx[j],:,:]                        # summed equivalent width
        table['summed_EW_mask_{}'.format(emline_names[j])][i] = maps['EMLINE_SEW_MASK'].data[emline_indx[j],:,:]            
        table['gaus_flux_{}'.format(emline_names[j])][i] = maps['EMLINE_GFLUX'].data[emline_indx[j],:,:]                      # gaussian flux
        table['gaus_ferr_{}'.format(emline_names[j])][i] = np.power(maps['EMLINE_GFLUX_IVAR'].data[emline_indx[j],:,:],-1/2)  # gaussian flux error
        table['gaus_flux_mask_{}'.format(emline_names[j])][i] = maps['EMLINE_GFLUX_MASK'].data[emline_indx[j],:,:]            # same mask for flux and error
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    