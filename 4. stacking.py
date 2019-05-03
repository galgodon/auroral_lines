# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 12:46:05 2019

@author: Gerome Algodon
"""
#%%
import time
t = time.clock()
# =============================================================================
# Import data
# =============================================================================
from astropy.io import fits
import numpy as np
#import matplotlib.pyplot as plt
import sys
import astropy.constants   # this is just used to get the speed of light c
c_vel = astropy.constants.c.to('km/s').value

spaxel_data_table = fits.open('data/spaxel_data_table.fits')

bin1,bin1_control = fits.open('data/Bin_1.fits'),fits.open('data/Bin_1_Control.fits')
bin2,bin2_control = fits.open('data/Bin_2.fits'),fits.open('data/Bin_2_Control.fits')
bin3,bin3_control = fits.open('data/Bin_3.fits'),fits.open('data/Bin_3_Control.fits')

flux_bin1, flux_bin1_control = fits.open('data/flux_bin_1.fits'), fits.open('data/flux_bin_1_control.fits')
flux_bin2, flux_bin2_control = fits.open('data/flux_bin_2.fits'), fits.open('data/flux_bin_2_control.fits')
flux_bin3, flux_bin3_control = fits.open('data/flux_bin_3.fits'), fits.open('data/flux_bin_3_control.fits')

def get_data(name,mask_name):
    return np.ma.MaskedArray(spaxel_data_table[1].data[name],mask=spaxel_data_table[1].data[mask_name]>0)

#spaxel_data_table[1].data.columns   # Use this to see column names
#%%
# =============================================================================
# Francesco's Drizzle Code    
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


#%%
# =============================================================================
# Stack
# =============================================================================
def drizzle(bin_num,flux_num,subbin_num):  # subbin_num: [x,y] x = bin number, y = subbin number
    ### Get wavelength array
    wave = flux_num[0].data
    wave_n = wave[wave < 8950]  # new wavelength array for the shifted data
    ### Pull data we need from the fits files
    z = (spaxel_data_table[1].data['z_vel'][bin_num['{}_{}'.format(subbin_num[0],subbin_num[1])].data.astype(int)])/c_vel
    stell_vel = get_data('stell_vel','stell_vel_mask')[bin_num['{}_{}'.format(subbin_num[0],subbin_num[1])].data.astype(int)]
    vel_off = bin_num['AVG_OFFSET_SUBBIN'].data[subbin_num[1]-1]
    #    ha_vel = get_data('ha_vel','ha_vel_mask')[bin_num['{}_{}'.format(subbin_num[0],subbin_num[1])].data.astype(int)]
    
    flux =  np.ma.MaskedArray(flux_num['FLUX_SUBBIN_{}'.format(subbin_num[1])].data,
                              mask=flux_num['MASK_SUBBIN_{}'.format(subbin_num[1])].data>0)
    var =  np.ma.power(np.ma.MaskedArray(flux_num['IVAR_SUBBIN_{}'.format(subbin_num[1])].data,
                              mask=flux_num['MASK_SUBBIN_{}'.format(subbin_num[1])].data>0),-1)
    
    flux_n = np.zeros(( len(flux) , len(wave_n) ))   # len(flux) is the nummber of spaxels in the bin
    f_weight = np.zeros(( len(flux) , len(wave_n) )) # and len(wave_n) is the length of the new wave array
    var_n = np.zeros(( len(flux) , len(wave_n) )) 
    v_weight = np.zeros(( len(flux) , len(wave_n) )) 
    
    for p in range(len(flux)):     
        zp = (1+(stell_vel[p] - vel_off)/c_vel)*(1+z[p])-1               # redshift at pixel 'p' note: ha_not_masked[p] = Velocity at pixel 'p'
        flux_n[p,:], f_weight[p,:] = specdrizzle_fast(wave, flux.data[p,:], zp, wave_n,mask=flux.mask[p,:].astype(int))
        var_n[p,:], v_weight[p,:] = specdrizzle_fast(wave, var.data[p,:], zp, wave_n,mask=var.mask[p,:].astype(int))
    
    flux_driz = np.ma.MaskedArray(flux_n, mask=f_weight==0)
    var_driz = np.ma.MaskedArray(var_n, mask=v_weight==0)
    return wave_n, flux_driz, var_driz

def stack(bin_num,flux_num,subbin_num):
    wave, flux_driz, var_driz = drizzle(bin_num,flux_num,subbin_num)
    flat = (wave>6000)&(wave<6100)
    med = np.median(flux_driz[:,flat].data,axis=1)
    flux_norm = flux_driz/med[:,None]
    var_norm = var_driz/(med[:,None]**2)
    
    totalflux_norm = np.ma.average(flux_norm,axis=0)
    totalvar_norm = np.ma.sum(var_norm,axis=0)/(np.sum(np.invert(var_norm.mask)))**2
    
    return wave, totalflux_norm, totalvar_norm

def get_bin_stack(num,bin_num,flux_num,name):
    print()
    print('Working on {}.fits'.format(name))
    stack_flux = np.zeros(( len(bin_num)-1 , 3930 ))  # 3930 = len(shortened wavelength from drizzle)
    stack_flux_mask = np.zeros(( len(bin_num)-1 , 3930 ),dtype='i4')
    
    stack_var = np.zeros(( len(bin_num)-1 , 3930 ))  
    stack_var_mask = np.zeros(( len(bin_num)-1 , 3930 ),dtype='i4')
    for i in range(len(bin_num)-1):   # the minus 1 is because the last indx in the fits file is just vel offset
        sys.stdout.write('\r'+('     Stacking sub-bin {}/{}'.format(i+1,len(bin_num)-1))) 
        wave, flux, var = stack(bin_num,flux_num,[num,i+1])
        stack_flux[i,:],stack_var[i,:] = flux.data, var.data
        stack_flux_mask[i,:],stack_var_mask[i,:] = flux.mask.astype(int), var.mask.astype(int)
    fit = fits.HDUList()
    fit.append(fits.PrimaryHDU(wave))
    fit.append(fits.ImageHDU(stack_flux,name='stacked_flux'))
    fit.append(fits.ImageHDU(stack_flux_mask,name='stacked_flux_mask'))
    fit.append(fits.ImageHDU(stack_var,name='stacked_var'))
    fit.append(fits.ImageHDU(stack_var_mask,name='stacked_var_mask'))
    fit.writeto('data/{}.fits'.format(name),overwrite=True)
    
    
#%%
get_bin_stack(1,bin1,flux_bin1,'bin_1_stack')
get_bin_stack(1,bin1_control,flux_bin1_control,'control_bin_1_stack')
get_bin_stack(2,bin2,flux_bin2,'bin_2_stack')
get_bin_stack(2,bin2_control,flux_bin2_control,'control_bin_2_stack')
get_bin_stack(3,bin3,flux_bin3,'bin_3_stack')
get_bin_stack(3,bin3_control,flux_bin3_control,'control_bin_3_stack')
#%% END
print('Code execution Time: {} sec'.format(time.clock()-t)) 
