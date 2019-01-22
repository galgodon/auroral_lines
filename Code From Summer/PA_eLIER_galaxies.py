# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 19:30:25 2018

@author: Gerome Algodon
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pafit.fit_kinematic_pa as findPA
from matplotlib import rc
from astropy.io import ascii
from astropy.table import Table
font = { 'size' : 6 }
fs = 10
rc('font', **font)

plt.ioff()

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

makegraphs = False

log = fits.open('data/MPL-6_master_catalogue_Apr9.fits')

eLIER = log[1].data['BPT_C'] == 3

indx = eLIER&np.invert((log[1].data['PLATE-IFU']=='8146-3702')|(log[1].data['PLATE-IFU']=='8158-3703'))
plate = log[1].data['PLATE'][indx]
ifu = log[1].data['IFUDESIGN'][indx]

drpall_file = '/data/manga/spectro/analysis/MPL-7/dapall-v2_4_3-2.2.1.fits'
drpall = fits.open(drpall_file)

angBestGM = np.array([])     # make empty arays
angErrGM = np.array([])
vSystGM = np.array([])

angBestSM = np.array([])
angErrSM = np.array([])
vSystSM = np.array([])

angDRP = np.array([])

print('{0:>10} {1:>15} {2:>10} {3:>15} {4:>10}'.format('Galaxy (Plate-IFU)','PA-GM', 'PA-SM', 'PA-DRPALL','Progress'))

snr_cutoff=6
spaxel_cutoff=30
k=0  # number of galaxies skipped
row = 4
col = 4
assert row*col%2==0,'Row \u00D7 Column must be even. {0}\u00D7{1} is not even'.format(row,col)  # Must be even for two graphs from the same galaxy to be next to each other
j=1
filenum=1
print('Set {0}'.format(filenum))
fig = plt.figure(figsize=(15,12))

for i in range(len(plate)):
    if (i%(((row*col)//2)) == 0)&(i !=0):
        plt.savefig('graphs_snr_{1}/Set_{0}.jpg'.format(filenum,snr_cutoff), format='jpg', dpi=300)
        filenum += 1
        print('Set {0}'.format(filenum))
        
        plt.close(fig)
        fig = plt.figure(figsize=(15,12))
        j=1
        
        
    maps_file = '/data/manga/spectro/analysis/MPL-7/HYB10-GAU-MILESHC/{0}/{1}/manga-{0}-{1}-MAPS-HYB10-GAU-MILESHC.fits.gz'.format(plate[i],ifu[i])
    maps = fits.open(maps_file)
    
    BA = drpall[1].data['NSA_ELPETRO_BA'][drpall[1].data['PLATEIFU'] == '{0}-{1}'.format(plate[i], ifu[i])][0]
    inc = 90 * (1-BA)
# =============================================================================
#     Compare calculated PA to listed PA in drpall
# =============================================================================
    l7 = drpall[1].data['NSA_ELPETRO_PHI'][drpall[1].data['PLATEIFU'] == '{0}-{1}'.format(plate[i], ifu[i])][0]
    angDRP = np.append(angDRP,l7)
# =============================================================================
#   finding position angle from h-alhpa velocity map (GM = gas map)
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
    
    nxGM, nyGM = ha_vel.shape
    pixGM = np.ma.MaskedArray(np.arange(nxGM).astype(float))
    xGM, yGM = np.meshgrid(pixGM, pixGM)
    
    ha_vel.mask = (ha_snr < snr_cutoff)|(np.abs(ha_vel)>800)|ha_vel.mask
    
    xGM[ha_vel.mask] = np.ma.masked
    yGM[ha_vel.mask] = np.ma.masked
    
    xGM = (xGM - len(xGM)/2)    
    yGM = (yGM - len(yGM)/2)
# =============================================================================
#   finding position angle from stellar velocity map (SM = star map)
# =============================================================================
    stellar_vel = np.ma.MaskedArray(maps['STELLAR_VEL'].data, mask=maps['STELLAR_VEL_MASK'].data > 0)
    stellar_vel_err = np.ma.power(np.ma.MaskedArray(maps['STELLAR_VEL_IVAR'].data,mask=maps['STELLAR_VEL_MASK'].data > 0), -0.5)
    stellar_snr = np.ma.MaskedArray(maps['SPX_SNR'].data, mask=maps['STELLAR_VEL_MASK'].data > 0)
    
    nxSM, nySM = stellar_vel.shape
    pixSM = np.ma.MaskedArray(np.arange(nxSM).astype(float))
    xSM, ySM = np.meshgrid(pixSM, pixSM)
    
    stellar_vel.mask = (stellar_snr < snr_cutoff)|(np.abs(stellar_vel)>800)|stellar_vel.mask
    
    xSM[stellar_vel.mask] = np.ma.masked
    ySM[stellar_vel.mask] = np.ma.masked
    
    xSM = (xSM - len(xSM)/2)     
    ySM = (ySM - len(ySM)/2)
# =============================================================================
#     
# =============================================================================
    if (len(ha_vel.compressed()) < spaxel_cutoff)|(len(stellar_vel.compressed()) < spaxel_cutoff):
        print('Galaxy {0}-{1} has < {2} spaxels remaining after masking signal to noise ratio < {3}'.format(plate[i],ifu[i],spaxel_cutoff,snr_cutoff))
        k += 1
        angBestGM = np.append(angBestGM,999999)
        angErrGM = np.append(angErrGM,999999)
        vSystGM = np.append(vSystGM,999999)
        angBestSM = np.append(angBestSM,999999)
        angErrSM = np.append(angErrSM,999999)
        vSystSM = np.append(vSystSM,999999)
        if makegraphs:
            plt.subplot(row,col,j)
            plt.imshow(ha_vel,origin='lower',interpolation='nearest', cmap='RdBu_r')
            plt.title('Ignore VM {0}-{1}'.format(plate[i],ifu[i]),fontsize=fs)
            cbar = plt.colorbar()    
            cbar.ax.set_ylabel('Velocity [km/s]', fontsize=fs)
            plt.subplot(row,col,j+1)
            plt.imshow(stellar_vel,origin='lower',interpolation='nearest', cmap='RdBu_r')
            plt.title('Ignore VM {0}-{1}'.format(plate[i],ifu[i]),fontsize=fs)
            cbar = plt.colorbar()    
            cbar.ax.set_ylabel('Velocity [km/s]', fontsize=fs)
            j+=2
        continue
# =============================================================================
#     
# =============================================================================
    
    l1,l2,l3 = findPA.fit_kinematic_pa(xGM.compressed(), yGM.compressed(), ha_vel.compressed(),quiet=True,plot=False)
    
    x1 = xGM*np.sin(np.radians(180-l1)) + yGM*np.cos(np.radians(180-l1))
    y1 = (yGM*np.sin(np.radians(180-l1)) - xGM*np.cos(np.radians(180-l1)))/np.cos(np.radians(inc))
    theta1 = np.degrees(np.arctan2(-y1, x1))
    wedge1 = 10
    indx1 = ((theta1 > -wedge1) & (theta1 < wedge1))    # CW part of wedge (part of wedge in right half)
    if np.mean(ha_vel[indx1]) > 0:                      # If its red, (positive velocity) add 180
        l1 += 180
    
    angBestGM = np.append(angBestGM,l1)
    angErrGM = np.append(angErrGM,l2)
    vSystGM = np.append(vSystGM,l3)
    
    if makegraphs:
        plt.subplot(row,col,j)
        plt.imshow(ha_vel,origin='lower',interpolation='nearest', cmap='RdBu_r')
        plt.title('H\u03B1 VM {0}-{1}'.format(plate[i],ifu[i]),fontsize=fs)
        cbar = plt.colorbar()    
        cbar.ax.set_ylabel('Velocity [km/s]', fontsize=fs)
        graphangle(l1,len(ha_vel)/2,l2)
        graphangle(l1+l2,len(ha_vel)/2,l2,dash=True)
        graphangle(l1-l2,len(ha_vel)/2,l2,dash=True)
        j += 1
# =============================================================================
# 
# =============================================================================
    
    l4,l5,l6 = findPA.fit_kinematic_pa(xSM.compressed(), ySM.compressed(), stellar_vel.compressed(),quiet=True,plot=False)
    
    x2 = xSM*np.sin(np.radians(180-l4)) + ySM*np.cos(np.radians(180-l4))
    y2 = (ySM*np.sin(np.radians(180-l4)) - xSM*np.cos(np.radians(180-l4)))/np.cos(np.radians(inc))
    theta2 = np.degrees(np.arctan2(-y2, x2))
    wedge2 = 10
    indx2 = ((theta2 > -wedge2) & (theta2 < wedge2))    # CW part of wedge (part of wedge in right half)
    if np.mean(stellar_vel[indx2]) > 0:                 # If its red, (positive velocity) add 180
        l4 += 180
    
    angBestSM = np.append(angBestSM,l4)
    angErrSM = np.append(angErrSM,l5)
    vSystSM = np.append(vSystSM,l6)
    
    if makegraphs:
        plt.subplot(row,col,j)
        plt.imshow(stellar_vel,origin='lower',interpolation='nearest', cmap='RdBu_r')
        plt.title('Stellar VM {0}-{1}'.format(plate[i],ifu[i]),fontsize=fs)
        cbar = plt.colorbar()    
        cbar.ax.set_ylabel('Velocity [km/s]', fontsize=fs)
        graphangle(l4,len(stellar_vel)/2,l5)
        graphangle(l4+l5,len(stellar_vel)/2,l5,dash=True)
        graphangle(l4-l5,len(stellar_vel)/2,l5,dash=True)
        j+=1
# =============================================================================
#   Print statement     
# =============================================================================
    print('{0:>18} {1:15.1f} {2:10.1f} {3:15.1f} {4:>10}'.format('{0}-{1}'.format(plate[i],ifu[i]),l1,l4,l7,'{0}/{1}'.format(i+1-k,len(plate)-k)))
    
    if (i==len(plate)-1):
        plt.savefig('graphs_snr_{1}/Set_{0}.jpg'.format(filenum,snr_cutoff), format='jpg', dpi=300)
data = Table([plate,ifu,angBestGM,angErrGM,vSystGM,angBestSM,angErrSM,vSystSM,angDRP],names=['Plate','IFU','PA_Ha','PA_Ha_Error','Ha_Vsyst','PA_Stellar','PA_Stellar_Error','Stellar_Vsyst','PA_drpall'])
ascii.write(data,'data/PA_eLIER_data_snr_{0}.txt'.format(snr_cutoff),overwrite=True)