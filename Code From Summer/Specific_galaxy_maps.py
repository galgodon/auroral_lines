# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 16:23:39 2018

@author: gerom
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pafit.fit_kinematic_pa as findPA
from matplotlib import rc
font = { 'size' : 6 }
fs = 10
rc('font', **font)

#spaxel_cutoff = int(input('Spaxel cutoff: '))
#snr_cutoff =  int(input('Signal to Noise Cutoff: '))
#PA = input('Do you want to calculate position angle? T/F  ')
#if not (PA == 'T')|(PA == 'F'):
#    print('Error, must be T or F')
#    exit()
spaxel_cutoff=30
snr_cutoff=6
PA = 'T'

plate = input('Plate: ')
ifu = input('IFU: ')
# =============================================================================
# Load Maps
# =============================================================================
drpall_file = '/data/manga/spectro/analysis/MPL-7/dapall-v2_4_3-2.2.1.fits'
drpall = fits.open(drpall_file)
maps_file = '/data/manga/spectro/analysis/MPL-7/HYB10-GAU-MILESHC/{0}/{1}/manga-{0}-{1}-MAPS-HYB10-GAU-MILESHC.fits.gz'.format(plate,ifu)
maps = fits.open(maps_file)

BA = drpall[1].data['NSA_ELPETRO_BA'][drpall[1].data['PLATEIFU'] == '{0}-{1}'.format(plate, ifu)][0]
inc = 90 * (1-BA)
# =============================================================================
# Pull gas map from maps
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
# =============================================================================
# pull stellar map from maps
# =============================================================================
stellar_vel = np.ma.MaskedArray(maps['STELLAR_VEL'].data, mask=maps['STELLAR_VEL_MASK'].data > 0)
stellar_vel_err = np.ma.power(np.ma.MaskedArray(maps['STELLAR_VEL_IVAR'].data,mask=maps['STELLAR_VEL_MASK'].data > 0), -0.5)
stellar_snr = np.ma.MaskedArray(maps['SPX_SNR'].data, mask=maps['STELLAR_VEL_MASK'].data > 0)

stellar_vel.mask = (stellar_snr < snr_cutoff)|(np.abs(stellar_vel)>800)|stellar_vel.mask
if PA == 'T':
    def graphangle(phi,x,err,dash=False):
        x1=-x*np.sin(np.radians(phi))+x+.5
        y1=x*np.cos(np.radians(phi))+x-.5
        x2=x*np.sin(np.radians(phi))+x+.5
        y2=-x*np.cos(np.radians(phi))+x-.5
        a = np.array([x1,x+.5,x2])
        b = np.array([y1,x-.5,y2])
        if dash:
            plt.plot(a,b,ls='dashed',c='k',alpha=0.73)
        else:
            plt.plot(a,b,c='k',label='PA={0}\u00B1{1}'.format(np.round(phi,2),np.round(err,2)))
            plt.legend()
    nxGM, nyGM = ha_vel.shape
    pixGM = np.ma.MaskedArray(np.arange(nxGM).astype(float))
    xGM, yGM = np.meshgrid(pixGM, pixGM)
    
    xGM[ha_vel.mask] = np.ma.masked
    yGM[ha_vel.mask] = np.ma.masked
    
    xGM = (xGM - len(xGM)/2)    
    yGM = (yGM - len(yGM)/2)
    
    nxSM, nySM = stellar_vel.shape
    pixSM = np.ma.MaskedArray(np.arange(nxSM).astype(float))
    xSM, ySM = np.meshgrid(pixSM, pixSM)
    
    stellar_vel.mask = (stellar_snr < snr_cutoff)|(np.abs(stellar_vel)>800)|stellar_vel.mask
    
    xSM[stellar_vel.mask] = np.ma.masked
    ySM[stellar_vel.mask] = np.ma.masked
    
    xSM = (xSM - len(xSM)/2)     
    ySM = (ySM - len(ySM)/2)
    
    if (len(ha_vel.compressed()) < spaxel_cutoff)|(len(stellar_vel.compressed()) < spaxel_cutoff):
        print('Galaxy {0}-{1} has < {2} spaxels remaining after masking signal to noise ratio < {3}'.format(plate,ifu,spaxel_cutoff,snr_cutoff))
        exit()

    l1,l2,l3 = findPA.fit_kinematic_pa(xGM.compressed(), yGM.compressed(), ha_vel.compressed(),quiet=True,plot=False)
    
    x1 = xGM*np.sin(np.radians(180-l1)) + yGM*np.cos(np.radians(180-l1))
    y1 = (yGM*np.sin(np.radians(180-l1)) - xGM*np.cos(np.radians(180-l1)))/np.cos(np.radians(inc))
    theta1 = np.degrees(np.arctan2(-y1, x1))
    wedge1 = 10
    indx1 = ((theta1 > -wedge1) & (theta1 < wedge1))    # CW part of wedge (part of wedge in right half)
    if np.mean(ha_vel[indx1]) > 0:                      # If its red, (positive velocity) add 180
        l1 += 180   

    l4,l5,l6 = findPA.fit_kinematic_pa(xSM.compressed(), ySM.compressed(), stellar_vel.compressed(),quiet=True,plot=False)
    
    x2 = xSM*np.sin(np.radians(180-l4)) + ySM*np.cos(np.radians(180-l4))
    y2 = (ySM*np.sin(np.radians(180-l4)) - xSM*np.cos(np.radians(180-l4)))/np.cos(np.radians(inc))
    theta2 = np.degrees(np.arctan2(-y2, x2))
    wedge2 = 10
    indx2 = ((theta2 > -wedge2) & (theta2 < wedge2))    # CW part of wedge (part of wedge in right half)
    if np.mean(stellar_vel[indx2]) > 0:                 # If its red, (positive velocity) add 180
        l4 += 180
        
    plt.figure(figsize=(20,10))
    
    plt.subplot(121)
    plt.imshow(ha_vel,origin='lower',interpolation='nearest', cmap='RdBu_r')
    plt.title('H\u03B1 VM {0}-{1}'.format(plate,ifu),fontsize=fs)
    cbar = plt.colorbar()    
    cbar.ax.set_ylabel('Velocity [km/s]', fontsize=fs)
    graphangle(l1,len(ha_vel)/2,l2)
    graphangle(l1+l2,len(ha_vel)/2,l2,dash=True)
    graphangle(l1-l2,len(ha_vel)/2,l2,dash=True)
    
    plt.subplot(122)
    plt.imshow(stellar_vel,origin='lower',interpolation='nearest', cmap='RdBu_r')
    plt.title('Stellar VM {0}-{1}'.format(plate,ifu),fontsize=fs)
    cbar = plt.colorbar()    
    cbar.ax.set_ylabel('Velocity [km/s]', fontsize=fs)
    graphangle(l4,len(stellar_vel)/2,l5)
    graphangle(l4+l5,len(stellar_vel)/2,l5,dash=True)
    graphangle(l4-l5,len(stellar_vel)/2,l5,dash=True)
    
    plt.show()
# =============================================================================
# Plot
# =============================================================================
if PA == 'F':
    plt.figure(figsize=(20,10))
    
    plt.subplot(121)
    plt.imshow(ha_vel,origin='lower',interpolation='nearest', cmap='RdBu_r')
    plt.title('H\u03B1 VM {0}-{1}'.format(plate,ifu),fontsize=fs)
    cbar = plt.colorbar()    
    cbar.ax.set_ylabel('Velocity [km/s]', fontsize=fs)
    
    plt.subplot(122)
    plt.imshow(stellar_vel,origin='lower',interpolation='nearest', cmap='RdBu_r')
    plt.title('Stellar VM {0}-{1}'.format(plate,ifu),fontsize=fs)
    cbar = plt.colorbar()    
    cbar.ax.set_ylabel('Velocity [km/s]', fontsize=fs)
    
    plt.show()