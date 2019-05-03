# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 13:50:31 2019

@author: gerom
"""

#%% Import packages
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

#%% Import data on stacks (created in 4. stacking.py)
stack_1 = fits.open('data/bin_1_stack.fits')
stack_2 = fits.open('data/bin_2_stack.fits')
stack_3 = fits.open('data/bin_3_stack.fits')

stack_1_control = fits.open('data/control_bin_1_stack.fits')
stack_2_control = fits.open('data/control_bin_2_stack.fits')
stack_3_control = fits.open('data/control_bin_3_stack.fits')

#%% Pull out strong and control for a specefic bin and subbin
def get_flat_resid(stack,stack_control,num):
    tot_bin_residual = np.zeros((25,3930)) # initialize array to accept all residual_new values
    for i in range(25):
        check_bin=i+1  # subbin
        bin_num = [stack,stack_control,num]  #bin_data
        
        wave = bin_num[0][0].data
        strong = np.ma.MaskedArray(bin_num[0]['STACKED_FLUX'].data[check_bin-1,:],
                                   mask=bin_num[0]['STACKED_FLUX_MASK'].data[check_bin-1,:].astype('bool'))
        control = np.ma.MaskedArray(bin_num[1]['STACKED_FLUX'].data[check_bin-1,:],
                                    mask=bin_num[0]['STACKED_FLUX_MASK'].data[check_bin-1,:].astype('bool'))
        
        ### Smooth out residual
        lines = np.array([3728.6,3836.485,3869.86,3889.750,3890.158,3933, 3971.202,    # copy-pasted from Renbin's code
                          4070,4078,4102.899,4305,4341.692,4862.691,4959,5007,5200.317,
                          5756.19,5877.249, 6300, 6549.86, 6564.632,6585.27, 6718.294,6732.674])
        lines = np.append(lines,np.array([3640,5420]))
        lineless = np.full(len(wave),True) # initialize array for lineless values
        width = np.full(len(lines),12)  # width around each line is 12, you can change specific width for specific line if needed
        width[lines==3640],width[lines==5420]=20,100
        
        for l in range(len(lines)):
            lineless[((wave>(lines[l]-width[l]))&(wave<(lines[l]+width[l])))]=False
        
        fit = np.polyfit(wave[lineless],strong[lineless]/control[lineless],3)
        fit_line = fit[0]*wave**3 + fit[1]*wave**2 + fit[2]*wave**1 + fit[3]*wave**0
#        
        residual_new = strong-control*fit_line  # this should be flat now
        tot_bin_residual[i,:]=residual_new

    return tot_bin_residual

bin_data_1 = fits.open('data/Bin_1.fits')
bin_data_2 = fits.open('data/Bin_2.fits')
bin_data_3 = fits.open('data/Bin_3.fits')

spx_num_1,spx_num_2,spx_num_3 = np.zeros(25),np.zeros(25),np.zeros(25)

for i in range(25):
    spx_num_1[i] = np.shape(bin_data_1[i].data)[0]
    spx_num_2[i] = np.shape(bin_data_2[i].data)[0]
    spx_num_3[i] = np.shape(bin_data_3[i].data)[0]
    
#%%
bin_1_resid = get_flat_resid(stack_1,stack_1_control,1)
bin_2_resid = get_flat_resid(stack_2,stack_2_control,2)
bin_3_resid = get_flat_resid(stack_3,stack_3_control,3)


bin1 = np.average(bin_1_resid,axis=0,weights=spx_num_1/spx_num_1.sum())
bin2 = np.average(bin_2_resid,axis=0,weights=spx_num_2/spx_num_2.sum())
bin3 = np.average(bin_3_resid,axis=0,weights=spx_num_3/spx_num_3.sum())

#%%
fs = 20  # fontsize for plots
wave = stack_1[0].data

#%% OIII
xlim =(4100,4600)

fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10,10))
fig.subplots_adjust(hspace=0)

axs[0].axvspan(4354.435,4374.435, color='gray', alpha=0.5,label='OIII-4363')  # values taken from renbin's paper table 2
axs[0].plot(wave, bin1,c='k')
axs[0].set_yticks(np.arange(-0.005, 0.0151, 0.005))
axs[0].set_ylim(-0.010,0.02)
axs[0].set_xlim(xlim)
axs[0].text(4150,0.012,r'High [NII]/H$\alpha$',fontsize=fs)
axs[0].legend(fontsize=fs-4)

axs[1].axvspan(4354.435,4374.435, color='gray', alpha=0.5)
axs[1].plot(wave, bin2,c='k')
axs[1].set_yticks(np.arange(-0.005, 0.0201, 0.005))
axs[1].set_ylim(-0.010,0.025)
axs[1].set_xlim(xlim)
axs[1].text(4150,0.014,r'Mid [NII]/H$\alpha$',fontsize=fs)
axs[1].set_ylabel(r'Rescaled Flux',fontsize=fs)

axs[2].axvspan(4354.435,4374.435, color='gray', alpha=0.5)
axs[2].plot(wave, bin3,c='k')
axs[2].set_yticks(np.arange(-0.005, 0.0251, 0.005))
axs[2].set_ylim(-0.010,0.030)
axs[2].set_xlim(xlim)
axs[2].text(4150,0.015,r'Low [NII]/H$\alpha$',fontsize=fs)
axs[2].set_xlabel(r'Rest Wavelength [$\AA$]',fontsize=fs)

plt.show()

#%% NII
xlim =(5500,6000)

fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10,10))
fig.subplots_adjust(hspace=0)

axs[0].axvspan(5746.2,5766.2, color='gray', alpha=0.5,label='NII-5755')
axs[0].plot(wave, bin1,c='k')
axs[0].set_yticks(np.arange(-0.006, 0.0061, 0.003))
axs[0].set_ylim(-0.007,0.007)
axs[0].set_xlim(xlim)
axs[0].text(5550,0.004,r'High [NII]/H$\alpha$',fontsize=fs)
axs[0].legend(fontsize=fs-4)

axs[1].axvspan(5746.2,5766.2, color='gray', alpha=0.5)
axs[1].plot(wave, bin2,c='k')
axs[1].set_yticks(np.arange(-0.003, 0.0061, 0.003))
axs[1].set_ylim(-0.007,0.01)
axs[1].set_xlim(xlim)
axs[1].text(5550,0.005,r'Mid [NII]/H$\alpha$',fontsize=fs)
axs[1].set_ylabel(r'Rescaled Flux',fontsize=fs)

axs[2].axvspan(5746.2,5766.2, color='gray', alpha=0.5)
axs[2].plot(wave, bin3,c='k')
axs[2].set_yticks(np.arange(-0.012, 0.0091, 0.003))
axs[2].set_ylim(-0.012,0.010)
axs[2].set_xlim(xlim)
axs[2].text(5550,0.006,r'Low [NII]/H$\alpha$',fontsize=fs)
axs[2].set_xlabel(r'Rest Wavelength [$\AA$]',fontsize=fs)

plt.show()

#%%OII
xlim =(7100,7600)

fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10,10))
fig.subplots_adjust(hspace=0)

axs[0].axvspan(7312.08,7342.08, color='gray', alpha=0.5,label='OII-7320,7330')
axs[0].plot(wave, bin1,c='k')
axs[0].set_yticks(np.arange(-0.004, 0.0081, 0.002))
axs[0].set_ylim(-0.005,0.010)
axs[0].set_xlim(xlim)
axs[0].text(7150,0.007,r'High [NII]/H$\alpha$',fontsize=fs)
axs[0].legend(fontsize=fs-4)

axs[1].axvspan(7312.08,7342.08, color='gray', alpha=0.5)
axs[1].plot(wave, bin2,c='k')
axs[1].set_yticks(np.arange(-0.004, 0.0081, 0.002))
axs[1].set_ylim(-0.005,0.010)
axs[1].set_xlim(xlim)
axs[1].text(7150,0.007,r'Mid [NII]/H$\alpha$',fontsize=fs)
axs[1].set_ylabel(r'Rescaled Flux',fontsize=fs)

axs[2].axvspan(7312.08,7342.08,  color='gray', alpha=0.5)
axs[2].plot(wave, bin3,c='k')
axs[2].set_yticks(np.arange(-0.004, 0.0081, 0.002))
axs[2].set_ylim(-0.005,0.010)
axs[2].set_xlim(xlim)
axs[2].text(7150,0.007,r'Low [NII]/H$\alpha$',fontsize=fs)
axs[2].set_xlabel(r'Rest Wavelength [$\AA$]',fontsize=fs)

plt.show()

#%% SII

xlim =(3900,4300)

fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10,10))
fig.subplots_adjust(hspace=0)

axs[0].axvspan(4058.625,4088.625, color='gray', alpha=0.5,label='SII-4068,4076')
axs[0].plot(wave, bin1,c='k')
axs[0].set_yticks(np.arange(-0.005, 0.015, 0.005))
axs[0].set_ylim(-0.010,0.015)
axs[0].set_xlim(xlim)
axs[0].text(3950,-0.006,r'High [NII]/H$\alpha$',fontsize=fs)
axs[0].legend(fontsize=fs-4)

axs[1].axvspan(4058.625,4088.625, color='gray', alpha=0.5)
axs[1].plot(wave, bin2,c='k')
axs[1].set_yticks(np.arange(-0.005, 0.015, 0.005))
axs[1].set_ylim(-0.01,0.015)
axs[1].set_xlim(xlim)
axs[1].text(3950,-0.006,r'Mid [NII]/H$\alpha$',fontsize=fs)
axs[1].set_ylabel(r'Rescaled Flux',fontsize=fs)

axs[2].axvspan(4058.625,4088.625, color='gray', alpha=0.5)
axs[2].plot(wave, bin3,c='k')
axs[2].set_yticks(np.arange(-0.005, 0.015, 0.005))
axs[2].set_ylim(-0.01,0.015)
axs[2].set_xlim(xlim)
axs[2].text(3950,-0.007,r'Low [NII]/H$\alpha$',fontsize=fs)
axs[2].set_xlabel(r'Rest Wavelength [$\AA$]',fontsize=fs)

plt.show()



