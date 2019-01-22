# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 08:56:04 2018

@author: gerom
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 20:19:02 2018

@author: Gerome Algodon
"""

import numpy as np
from astropy.io import fits
import astropy.constants
from astropy.io import ascii
from astropy.table import Table
# =============================================================================
# 
# =============================================================================
def _specbin( wl):
  """ given wavelength vector wl, return wl_lo and wl_hi limits of wavelength bins"""
  nwl = len(wl)

  dwl_lo = wl-np.roll(wl, 1)
  dwl_lo[0] = dwl_lo[1]

  dwl_hi = np.roll(wl, -1)-wl
  dwl_hi[nwl-1] = dwl_hi[nwl-2]

#  find limits of spectral bins
  wl_lo = wl-dwl_lo/2.0
  wl_hi = wl+dwl_hi/2.0
  return wl_lo, wl_hi

def specdrizzle_fast(wl, spec, z, wldrz, mask=None, flux=True):
#"""
# Drizzles the spectrum into a new rest frame wavelength vector
# Fully support masks
#    THIS IS THE FASTER VERSION INSPIRED BY Carnall 2017
# 
# wl    - input wavelength
# spec  - input spectrum
# z     - redshift
# wldrz - output rest frame wavelength
# spdrz - output drizzled spectrum
# spwgt - output weight
# mask  - mask of good pixels (value=1)
# flux  - 1 total flux of the spectrum is conserved in the drizzling procedure
#"""

    #     length of input and output wavelength vectors
#    nwl = len(wl)
    nwldrz = len(wldrz)
     
    if nwldrz < 2:
        raise ValueError('output wavelength grid must have at least 2 elements')
    
    #    initialize output spectrum
    spdrz = np.zeros(len(wldrz))
    spwgt = np.zeros(len(wldrz))
    
    #    all pixels good if mask not defined
    if mask is None:
        mask = spec*0.
        print('not masking')
        
    if flux==True: 
    #    conserve flux after redshift correction
        specz = spec*(1.0+z)
#        print 'conserving flux'
    else:
        specz = spec

    mask2=np.where(mask==0, 1, 0)
    
    wlz1 = np.zeros(wl.shape[0])
    dwlz = np.zeros(wl.shape[0])
    
    wl=wl/(1.0+z)

    wlz1, wl_hi= _specbin(wl)
    dwlz = wl_hi - wlz1
  
    wldrz1, wldrz_hi= _specbin(wldrz)
    dwldrz= wldrz_hi - wldrz1
    
    # Calculate the new spectral flux and uncertainty values, loop over the new bins
    for j in range(wldrz.shape[0]):
        
        # Find the first old bin which is partially covered by the new bin
#        while wlz1[start+1] <= wldrz1[j]:
#            start += 1
            
        start_v = np.where(wl_hi > wldrz1[j])[0]
        if len(start_v) == 0:
            spdrz[j]=0.
            spwgt[j] =0
            continue
        else:
            start=start_v[0]
        # Find the last old bin which is partially covered by the new bin
#        while (stop <= len(wlz1)) and (wlz1[stop+1] < wldrz1[j+1]):
#            stop += 1
            
        stop = np.where(wlz1 <= wldrz_hi[j])[0][-1]
        
#        print start, stop
#
#        If the new bin falls entirely within one old bin they are the same 
#        the new flux and new error are the same as for that bin
        if stop == start:

            spdrz[j]= specz[start]*mask2[start]
            spwgt[j] = mask2[start]
      
          
        # Otherwise multiply the first and last old bin widths by P_ij, 
#        all the ones in between have P_ij = 1 
        else:

            start_factor = (wl_hi[start] - wldrz1[j])/(dwlz[start])
            end_factor = (wldrz_hi[j] - wlz1[stop])/(dwlz[stop])

            dwlz[start] *= start_factor
            dwlz[stop] *= end_factor

            # Populate the resampled_fluxes spectrum and uncertainty arrays
            
            spwgt[j] = np.sum(dwlz[start:stop+1]*mask2[start:stop+1])/dwldrz[j]
            if spwgt[j]>0:
                spdrz[j] = np.sum(dwlz[start:stop+1]*specz[start:stop+1]*mask2[start:stop+1])/  \
                   np.sum(dwlz[start:stop+1]*mask2[start:stop+1])
            

#            if spec_errs is not None:
#                resampled_fluxes_errs[...,j] = np.sqrt(np.sum((spec_widths[start:stop+1]*spec_errs[...,start:stop+1])**2, axis=-1))/np.sum(spec_widths[start:stop+1])
#            
            # Put back the old bin widths to their initial values for later use
            dwlz[start] /= start_factor
            dwlz[stop] /= end_factor

    
    
    return spdrz, spwgt
# =============================================================================
# 
# =============================================================================
alignthreshold = 30
spaxel_cutoff=30
snr_cutoff=6

c = astropy.constants.c.to('km/s').value

drpall_file = '/data/manga/spectro/analysis/MPL-7/dapall-v2_4_3-2.2.1.fits'
drpall = fits.open(drpall_file)
for sn in range(4):
    stack_number = sn+1
#    stack_number = 'test'
    assert (stack_number == 1)|(stack_number == 2)|(stack_number == 3)|(stack_number == 4)|(stack_number=='test')
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
        
        logcube_file = '/data/manga/spectro/redux/MPL-7/{0}/stack/manga-{0}-{1}-LOGCUBE.fits.gz'.format(plate[i],ifu[i])
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
        weight = np.empty_like(flux_c[:wavecut.sum()])
        
        for p in range(len(ha_not_masked)):     
            zp = (1+ha_not_masked[p]/c)*(1+z)-1               # redshift at pixel 'p' note: ha_not_masked[p] = Velocity at pixel 'p'
            flux_n[:,p], weight[:,p] = specdrizzle_fast(wave, flux_c.data[:,p], zp, wave[wavecut],mask=flux_c.mask[:,p].astype(int))
        
        _flux = np.ma.MaskedArray(flux_n, mask=weight==0)
        
        totalflux = np.ma.average(_flux,axis=1)
        totalflux_ha_w = np.ma.average(_flux,axis=1,weights=(ha_flux[~ha_vel.mask]/np.sum(ha_flux[~ha_vel.mask])))
        totalflux_cont_snr_w = np.ma.average(_flux,axis=1,weights=stellar_snr[~ha_vel.mask]/np.sum(stellar_snr[~ha_vel.mask]))
        
        flux_sums = np.zeros((len(totalflux), 3))
        flux_sums[:,0] = totalflux.data
        flux_sums[:,1] = totalflux_ha_w.data
        flux_sums[:,2] = totalflux_cont_snr_w.data
        
        hdul.append(fits.ImageHDU(flux_sums,name='{0}-{1}'.format(plate[i],ifu[i])))
        hdul.append(fits.ImageHDU(weight.data,name='{0}-{1}_mask'.format(plate[i],ifu[i])))
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