# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 11:41:24 2019

@author: Gerome Algodon

Creating table of data for Winter 2019 Research project - Finding Auroral Lines
"""

import numpy as np
from astropy.io import fits
from astropy.table import Table,vstack
import os

import sys

import warnings                    # I put this here so the invalid value in sqrt warning would be be ignored
warnings.filterwarnings("ignore")

# =============================================================================
# Start with an array of Plate-Ifu's to make the table for
# =============================================================================

# using the plate and ifu that I found in galaxy_selection.py

log = fits.open('data/quiescent_red_sequence_galaxies.fits')

plate = log[1].data['plate']
ifu = log[1].data['ifudsgn']

#plate = np.array([7977,8143]) TEST
#ifu = np.array([12705,3701])

# =============================================================================
# Check that all galaxies are in MPL-7, if they are not print them and stop code
# =============================================================================

for i in range(len(plate)):
    maps_file = '/data/manga/spectro/analysis/MPL-7/HYB10-GAU-MILESHC/{0}/{1}/manga-{0}-{1}-MAPS-HYB10-GAU-MILESHC.fits.gz'.format(plate[i],ifu[i])
    if not os.path.isfile(maps_file):
        sys.exit('Galaxy plate-ifu: {}-{} not found in MPL-7'.format(plate[i],ifu[i]))
        
# =============================================================================
# Initialize Columns for table
# =============================================================================

names = ['plate','ifu','z_vel','gal_red_B-V','stell_vel_ID','emline_ID',
         'rad_norm_spx','azimuth_spx','snr_spx','snr_bin','stell_vel',
         'stell_vel_mask','ha_vel','ha_vel_mask','stell_sigma_cor',
         'stell_sigma_mask','spec_index_D4000','spec_index_ivar_D4000',
         'spec_index_mask_D4000','gmr','gmr_mask','flux_r_band','gmr_corr','spec_index_HDeltaA',
         'spec_index_ivar_HDeltaA','spec_index_mask_HDeltaA']
dtype = ['i4','i4','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8',
         'f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8']
assert len(names)==len(dtype)
emline_names = np.array(['H_alpha','H_beta','OII-3727','OII-3729','NII-6585','SII-6718',
                         'SII-6732','OIII-4960','OIII-5008','OI-6302','OI-6365'])
emline_indx = np.array([18,11,0,1,19,20,21,12,13,15,16])  # corresponding index numbers found on https://trac.sdss.org/wiki/MANGA/TRM/TRM_MPL-7/DAPDataModel#DAPMAPSfile
assert len(emline_names)==len(emline_indx)
for k in range(len(emline_names)):
    names.append('summed_EW_{}'.format(emline_names[k]))
    names.append('summed_EW_IVAR_{}'.format(emline_names[k]))
    names.append('summed_EW_mask_{}'.format(emline_names[k]))
    
    names.append('gauss_EW_{}'.format(emline_names[k]))
    names.append('gauss_EW_IVAR_{}'.format(emline_names[k]))
    names.append('gauss_EW_mask_{}'.format(emline_names[k]))
    
    names.append('gaus_flux_{}'.format(emline_names[k]))
    names.append('gaus_flux_IVAR_{}'.format(emline_names[k]))
    names.append('gaus_flux_mask_{}'.format(emline_names[k]))
    dtype.append('f8')
    dtype.append('f8')
    dtype.append('f8')
    dtype.append('f8')
    dtype.append('f8')
    dtype.append('f8')
    dtype.append('f8')
    dtype.append('f8')
    dtype.append('f8')
table = Table(names=names,dtype=dtype)

# =============================================================================
# Now we need to pull all of the data we need for each spaxel in each galaxy
# =============================================================================

def gmr_data(drp_file,gs):  # Code I got from Kyle to get g-r data from logcube file
    # gs = good_spaxel mask that I use below
    if not os.path.isfile(drp_file):
        raise FileNotFoundError('{0} does not exist!'.format(drp_file))
    hdu = fits.open(drp_file)
    gmr_map = -2.5*np.ma.log10(np.ma.MaskedArray(hdu['GIMG'].data,mask=np.invert(hdu['GIMG'].data>0)) 
                / np.ma.MaskedArray(hdu['RIMG'].data,mask=np.invert(hdu['RIMG'].data>0)))
    r = np.ma.MaskedArray(hdu['RIMG'].data,mask=np.invert(hdu['RIMG'].data>0))
    hdu.close()
    del hdu
    return gmr_map.data.flatten()[gs], gmr_map.mask.flatten()[gs], r.data.flatten()[gs]

drp_all = fits.open('/data/manga/spectro/redux/MPL-7/drpall-v2_4_3.fits')
plate_ifu_drp = drp_all[1].data['plateifu']
g = drp_all[1].data['nsa_elpetro_absmag'][:,3]  # fnugriz
r = drp_all[1].data['nsa_elpetro_absmag'][:,4]
g_r = g - r
g_ob = 22.5-2.5*np.log10(drp_all[1].data['nsa_elpetro_flux'][:,3])
r_ob = 22.5-2.5*np.log10(drp_all[1].data['nsa_elpetro_flux'][:,4])
g_r_ob = g_ob - r_ob

def get_gmr_corr(plate,ifu):   # get the gmr
    gal = plate_ifu_drp==('{}-{}'.format(plate,ifu))
    assert gal.sum()==1,'{}-{}'.format(plate,ifu)
    return g_r_ob[gal] - g_r[gal]
    
for i in range(len(plate)):
    
    sys.stdout.write('\r'+('Galaxy {}/{}'.format(i+1,len(plate))))  # print statement to visually check progress that deletes and replaces itself
    
    maps_file = '/data/manga/spectro/analysis/MPL-7/HYB10-GAU-MILESHC/{0}/{1}/manga-{0}-{1}-MAPS-HYB10-GAU-MILESHC.fits.gz'.format(plate[i],ifu[i])  # load in maps file for this galaxy
    logcube_file = '/data/manga/spectro/redux/MPL-7/{0}/stack/manga-{0}-{1}-LOGCUBE.fits.gz'.format(plate[i],ifu[i])
    maps = fits.open(maps_file)
    
#    maps_file = 'data/manga-{0}-{1}-MAPS-HYB10-GAU-MILESHC.fits.gz'.format(plate[i],ifu[i])  # TEST
#    logcube_file = 'data/manga-{0}-{1}-LOGCUBE.fits.gz'.format(plate[i],ifu[i])
#    maps = fits.open(maps_file)
    
    good_spaxel = (maps['BINID'].data[0,:,:].flatten() != -1)   # ignore useless spaxels (like the corners)

    row = np.zeros((good_spaxel.sum(),len(dtype)))  # initialize array where we will be storing row data
    
    # Get Plate and IFU
    row[:,0] = (np.full(good_spaxel.sum(),plate[i]))  # plate and ifu is the same for every spaxel which is why I use np.full     
    row[:,1] = (np.full(good_spaxel.sum(),ifu[i]))
    
    # Get z_vel and gal_red_B-V
    row[:,2] = (np.full(good_spaxel.sum(),maps[0].header['SCINPVEL']))
    row[:,3] = (np.full(good_spaxel.sum(),maps[0].header['EBVGAL']))

    # Get Bin IDs we want
    row[:,4] = (maps['BINID'].data[1,:,:].flatten()[good_spaxel])  # stellar
    row[:,5] = (maps['BINID'].data[3,:,:].flatten()[good_spaxel])  # emline 1 bin per spaxel

    # Get coordinates
    row[:,6] = (maps['SPX_ELLCOO'].data[1,:,:].flatten()[good_spaxel])  # radius normalized by the elliptical Petrosian effective radius from the NSA
    row[:,7] = (maps['SPX_ELLCOO'].data[2,:,:].flatten()[good_spaxel])  # azimuth angle
    
    # Get continuum signal to noise ratio
    row[:,8] = (maps['SPX_SNR'].data.flatten()[good_spaxel])  # S/N in each spaxel (emission-line measurements done per spaxel)
    row[:,9] = (maps['BIN_SNR'].data.flatten()[good_spaxel])  # S/N in each bin (stellar measurements done per bin)
    
    # Get stellar velocity
    row[:,10] = (maps['STELLAR_VEL'].data.flatten()[good_spaxel])
    row[:,11] = (maps['STELLAR_VEL_MASK'].data.flatten()[good_spaxel])
    
    # Get H_alpha velocity (indx 18, channel 19)
    row[:,12] = (maps['EMLINE_GVEL'].data[18,:,:].flatten()[good_spaxel]) 
    row[:,13] = (maps['EMLINE_GVEL_MASK'].data[18,:,:].flatten()[good_spaxel])

    # Get corrected velocity dispersion
    row[:,14] = ((np.sqrt( (maps['STELLAR_SIGMA'].data)**2 - (maps['STELLAR_SIGMACORR'].data)**2 )).flatten()[good_spaxel])
    row[:,15] = (maps['STELLAR_SIGMA_MASK'].data.flatten()[good_spaxel])
    
    # Get spectral index for D4000 (indx 43, channel 44)
    row[:,16] = (maps['SPECINDEX'].data[43,:,:].flatten()[good_spaxel])
    row[:,17] = (maps['SPECINDEX_IVAR'].data[43,:,:].flatten()[good_spaxel])
    row[:,18] = (maps['SPECINDEX_MASK'].data[43,:,:].flatten()[good_spaxel])
    
    # Get g-r data using Kyle's code. 
    row[:,19],row[:,20],row[:,21] = gmr_data(logcube_file,good_spaxel)
    row[:,22] = np.full(good_spaxel.sum(),get_gmr_corr(plate[i],ifu[i]))  # same corr for every spaxel

    # Get spectral index for HDeltaA (indx 21, channel 22) note: this has a correction but D4000 didn't
    row[:,23] = (maps['SPECINDEX'].data[21,:,:].flatten()*maps['SPECINDEX_CORR'].data[21,:,:].flatten())[good_spaxel]
    row[:,24] = (maps['SPECINDEX_IVAR'].data[21,:,:].flatten()[good_spaxel])
    row[:,25] = (maps['SPECINDEX_MASK'].data[21,:,:].flatten()[good_spaxel])
    
    # Get emmision line properties of Ha, hb, o2, n2, s2, o3, o1
    
    for j in range(len(emline_names)):
        row[:,26+9*j] = (maps['EMLINE_SEW'].data[emline_indx[j],:,:].flatten()[good_spaxel])          # summed equivalent width
        row[:,27+9*j] = (maps['EMLINE_SEW_IVAR'].data[emline_indx[j],:,:].flatten()[good_spaxel])  
        row[:,28+9*j] = (maps['EMLINE_SEW_MASK'].data[emline_indx[j],:,:].flatten()[good_spaxel])  
        
        row[:,29+9*j] = (maps['EMLINE_GEW'].data[emline_indx[j],:,:].flatten()[good_spaxel])          # gauss equivalent width
        row[:,30+9*j] = (maps['EMLINE_GEW_IVAR'].data[emline_indx[j],:,:].flatten()[good_spaxel])  
        row[:,31+9*j] = (maps['EMLINE_GEW_MASK'].data[emline_indx[j],:,:].flatten()[good_spaxel])      
        
        row[:,32+9*j] = (maps['EMLINE_GFLUX'].data[emline_indx[j],:,:].flatten()[good_spaxel])        # gaussian flux
        row[:,33+9*j] = (maps['EMLINE_GFLUX_IVAR'].data[emline_indx[j],:,:].flatten()[good_spaxel])   # gaussian flux error
        row[:,34+9*j] = (maps['EMLINE_GFLUX_MASK'].data[emline_indx[j],:,:].flatten()[good_spaxel])   # same mask for flux and error

    table = vstack([table,Table(row,names=names,dtype=dtype)])
    
table.write('data/spaxel_data_table.fits', format='fits',overwrite=True)