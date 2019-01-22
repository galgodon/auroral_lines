# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 20:19:02 2018

@author: Gerome Algodon
"""

import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d
import astropy.constants
from astropy.io import ascii
from astropy.table import Table

for sn in range(4):
    stack_number = sn+1
#    stack_number = 'test'
    assert (stack_number == 1)|(stack_number == 2)|(stack_number == 3)|(stack_number == 4)|(stack_number=='test')
    
    alignthreshold = 30
    spaxel_cutoff=30
    snr_cutoff=6
    
    c = astropy.constants.c.to('km/s').value
    
    drpall_file = '/data/manga/spectro/analysis/MPL-7/dapall-v2_4_3-2.2.1.fits'
    drpall = fits.open(drpall_file)
    # =============================================================================
    # 
    # =============================================================================
    data = ascii.read('data/PA_eLIER_data_snr_{0}.txt'.format(snr_cutoff))
     
    gasPA_all = data['PA_Ha'].data
    stellarPA_all = data['PA_Stellar'].data
    gasPA_err = data['PA_Ha_Error'].data
    stellarPA_err = data['PA_Stellar_Error'].data
    err_cutoff = 30
    
    good_data=(gasPA_err<err_cutoff)&(stellarPA_err<err_cutoff)
    
    gasPA = gasPA_all[good_data]
    stellarPA = stellarPA_all[good_data]
    
    deltaPA = np.abs(stellarPA-gasPA)
    for i in range(len(deltaPA)):
        if deltaPA[i] > 180:
            deltaPA[i] = 360 - deltaPA[i]
    
    aligned = deltaPA < alignthreshold
    misaligned = np.invert(aligned)
    
    log = fits.open('data/MPL-6_master_catalogue_Apr9.fits')
    
    eLIER = log[1].data['BPT_C'] == 3
    
    indx = eLIER&np.invert((log[1].data['PLATE-IFU']=='8146-3702')|(log[1].data['PLATE-IFU']=='8158-3703'))
    
    mass = log[1].data['MASS_ELL_PETRO'][indx][good_data]
    
    center_mass = np.median(mass)
    low_mass = mass <= center_mass
    high_mass = mass > center_mass
    
    aligned_low = deltaPA[low_mass] < alignthreshold
    misaligned_low = np.invert(aligned_low)
    
    aligned_high = deltaPA[high_mass] < alignthreshold
    misaligned_high = np.invert(aligned_high)
    hdul = fits.HDUList()
    if stack_number == 2:
        hdul.append(fits.PrimaryHDU(wave))
    # =============================================================================
    # 
    # =============================================================================
    if stack_number == 1:
        plate = log[1].data['PLATE'][indx][good_data][low_mass][aligned_low]
        ifu = log[1].data['IFUDESIGN'][indx][good_data][low_mass][aligned_low]
    
    if stack_number == 2:
        plate = log[1].data['PLATE'][indx][good_data][low_mass][misaligned_low]
        ifu = log[1].data['IFUDESIGN'][indx][good_data][low_mass][misaligned_low]
    
    if stack_number ==3:
        plate = log[1].data['PLATE'][indx][good_data][high_mass][aligned_high]
        ifu = log[1].data['IFUDESIGN'][indx][good_data][high_mass][aligned_high]
    
    if stack_number == 4:
        plate = log[1].data['PLATE'][indx][good_data][high_mass][misaligned_high]
        ifu = log[1].data['IFUDESIGN'][indx][good_data][high_mass][misaligned_high]
        
    if stack_number == 'test':
        plate = np.array([8138])
        ifu = np.array([12703])
    # =============================================================================
    # 
    # =============================================================================
    for i in range(len(plate)):
        maps_file = '/data/manga/spectro/analysis/MPL-7/HYB10-GAU-MILESHC/{0}/{1}/manga-{0}-{1}-MAPS-HYB10-GAU-MILESHC.fits.gz'.format(plate[i],ifu[i])
        maps = fits.open(maps_file)
        
        logcube_file = '/data/manga/spectro/analysis/MPL-7/HYB10-GAU-MILESHC/{0}/{1}/manga-{0}-{1}-LOGCUBE-HYB10-GAU-MILESHC.fits.gz'.format(plate[i],ifu[i])
        log = fits.open(logcube_file)
        
        z = drpall[1].data['NSA_Z'][drpall[1].data['PLATEIFU']=='{0}-{1}'.format(plate[i],ifu[i])][0]
        # =============================================================================
        # Pull ha vel from maps_file
        # =============================================================================
        emline_vel = np.ma.MaskedArray(maps['EMLINE_GVEL'].data, mask=maps['EMLINE_GVEL_MASK'].data > 0)
        emline_vel_err = np.ma.power(np.ma.MaskedArray(maps['EMLINE_GVEL_IVAR'].data,mask=maps['EMLINE_GVEL_MASK'].data > 0), -0.5)
        emline_flux = np.ma.MaskedArray(maps['EMLINE_GFLUX'].data, mask=maps['EMLINE_GFLUX_MASK'].data > 0)
        emline_flux_err = np.ma.power(np.ma.MaskedArray(maps['EMLINE_GFLUX_IVAR'].data,mask=maps['EMLINE_GFLUX_MASK'].data > 0), -0.5)
        
        ha_vel = emline_vel[18,:,:]     #h-alpha is in channel 19, so indx 18. But note that all channels in this array are identical so it doesn't technically matter.
        ha_vel_err = emline_vel_err[18,:,:]
        ha_flux = emline_flux[18,:,:]
        ha_flux_err = emline_flux_err[18,:,:]
        ha_snr = ha_flux/ha_flux_err
        
        ha_vel.mask = (ha_snr < snr_cutoff)|(np.abs(ha_vel)>800)|ha_vel.mask
        
        stellar_snr = np.ma.MaskedArray(maps['SPX_SNR'].data, mask=maps['STELLAR_VEL_MASK'].data > 0)
        # =============================================================================
        # get original wave and flux from logcube_file
        # =============================================================================
        wave = log['WAVE'].data
        flux = np.ma.MaskedArray(log['FLUX'].data, mask=log['MASK'].data > 0)
        # =============================================================================
        # define where to cutoff wavelength after everything has been shifted down
        # =============================================================================
        vmax = np.max(ha_vel)
        wavemax = np.max(wave/(1+(vmax/c))/(1+z))
        wavecut = wave < wavemax
        # =============================================================================
        # Create new flux array shifted to rest wavelength
        # =============================================================================
        ha_not_masked = ha_vel[~ha_vel.mask]            # Velocity at each 'good pixel'
                                                        # Number of good pixels = ha_not_masked.shape
        flux_c = flux[:, ~ha_vel.mask]                  # Copy of flux at each 'good pixel'
        
        flux_n = np.empty_like(flux_c[:wavecut.sum()].filled())  # Empty array for new flux 
        
        for j in range(len(ha_not_masked)):
            vel = ha_not_masked[j]                      # Velocity at pixel 'j'
            wave_rest = wave/(1+(vel/c))/(1+z)          # Shifting wavelength to rest wavelength using z,vel,and c
            f_int = interp1d(wave_rest,flux_c[:,j])     # Interpolate flux values so that we can...
            flux_n[:,j] = f_int(wave[wavecut])          # Make a new flux based on the original wavelength grid 
                                                        # and insert it into new var flux_n
#            flux_n[:,j],w = specdrizzle_fast(wave_rest,flux_c[:,j],z,wave[wavecut],mask=flux_c[:,j].mask,flux=False)
        
        totalflux = np.average(flux_n,axis=1)
        totalflux_ha_w = np.average(flux_n,axis=1,weights=ha_flux[~ha_vel.mask])
        totalflux_cont_snr_w = np.average(flux_n,axis=1,weights=stellar_snr[~ha_vel.mask])
        
        flux_sums = np.zeros((len(totalflux), 3))
        flux_sums[:,0] = totalflux
        flux_sums[:,1] = totalflux_ha_w
        flux_sums[:,2] = totalflux_cont_snr_w
        
        hdul.append(fits.ImageHDU(flux_sums,name='{0}-{1}'.format(plate[i],ifu[i])))
        print('Progress: {0}/{1}'.format(i+1,len(plate)))
    # =============================================================================
    # 
    # =============================================================================
    if stack_number == 1:
        hdul.writeto('data/fluxtot_low_mass_aligned.fits',overwrite=True)
        data = Table([plate,ifu],names=['Plate','IFU'])
        ascii.write(data,'data/fluxtot_low_mass_aligned.txt',overwrite=True)
    if stack_number == 2:
        hdul.writeto('data/fluxtot_low_mass_misaligned.fits',overwrite=True)
        data = Table([plate,ifu],names=['Plate','IFU'])
        ascii.write(data,'data/fluxtot_low_mass_misaligned.txt',overwrite=True)
    if stack_number == 3:
        hdul.writeto('data/fluxtot_high_mass_aligned.fits',overwrite=True)
        data = Table([plate,ifu],names=['Plate','IFU'])
        ascii.write(data,'data/fluxtot_high_mass_aligned.txt',overwrite=True)
    if stack_number == 4:
        hdul.writeto('data/fluxtot_high_mass_misaligned.fits',overwrite=True)
        data = Table([plate,ifu],names=['Plate','IFU'])
        ascii.write(data,'data/fluxtot_high_mass_misaligned.txt',overwrite=True)
    if stack_number == 'test':
        hdul.writeto('test/test.fits',overwrite=True)
        data = Table([plate,ifu],names=['Plate','IFU'])
        ascii.write(data,'test/test.txt',overwrite=True)
    # =============================================================================
    # 
    # =============================================================================
    print('End of stack {0}'.format(stack_number))