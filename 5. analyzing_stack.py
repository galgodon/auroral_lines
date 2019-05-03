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

check_bin=13  # subbin
bin_num = [stack_3,stack_3_control,3]  #bin_data

wave = bin_num[0][0].data
strong = bin_num[0]['STACKED_FLUX'].data[check_bin-1,:]
control = bin_num[1]['STACKED_FLUX'].data[check_bin-1,:]

residual = strong - control
#"""
#%% Make two panel plot. Panel 1: strong and control on the same plot. 
### Panel 2: strong - control (residual)
plt.figure(figsize=(20,7))

plt.subplot(1,2,1)
plt.plot(wave,strong,label='Bin 1')
plt.plot(wave,control,label='Control Bin 1')
plt.xlim((wave[0],wave[-1]))
plt.title('Stacked spectra: Bin {} / Sub-bin {}'.format(bin_num[2],check_bin))
plt.legend()
plt.show()

plt.subplot(1,2,2)
plt.plot(wave,residual)
plt.xlim((wave[0],wave[-1]))
plt.title('Bin {1}.{0} - Control Bin {1}.{0}'.format(check_bin,bin_num[2]))
#plt.ylim(-0.7,0.3)
plt.show()

#%% Checking specific wavelengths for strong line-control
# 3969 3934  stellar Ca H K lines after change these shouldn't be centered but gas should
# Ha = 6564, OIII = 5008
plt.figure()
check = 6564  # check this wavelength
p_m = 200  # window will be check+/-p_m
plt.plot(wave,residual)
plt.axvline(check,label=r'{} $\AA$'.format(check),ls='dashed',c='k')
plt.xlim((check-p_m,check+p_m))  
plt.title('Bin {1}.{0} - Control Bin {1}.{0}'.format(check_bin,bin_num[2]))
plt.legend()
plt.show()

#%% Looking for auroral lines

plt.figure(figsize=(20,20))

plt.subplot(2,2,1)
plt.plot(wave,residual)
plt.axvline(4363,label='OIII-4363',ls='dashed',c='k')
plt.xlim((4100,4600))  # lim from renbin's paper fig 10
plt.ylim((-0.10,0.05))
plt.title('Bin {1}.{0} - Control Bin {1}.{0}'.format(check_bin,bin_num[2]))
plt.legend()
plt.show()

plt.subplot(2,2,2)
plt.plot(wave,residual)
plt.axvline(5755,label='NII-5755',ls='dashed',c='k')
plt.xlim((5500,6000))  # lim from renbin's paper fig 10
plt.ylim((-0.1,0.1))
plt.title('Bin {1}.{0} - Control Bin {1}.{0}'.format(check_bin,bin_num[2]))
plt.legend()
plt.show()

plt.subplot(2,2,3)
plt.plot(wave,residual)
plt.axvline(7320,label='OII-7320,7330',ls='dashed',c='k')
plt.axvline(7330,ls='dashed',c='k')
plt.xlim((7100,7600))  # lim from renbin's paper fig 10
plt.ylim((-0.05,0.3))
plt.title('Bin {1}.{0} - Control Bin {1}.{0}'.format(check_bin,bin_num[2]))
plt.legend()
plt.show()

plt.subplot(2,2,4)
plt.plot(wave,residual)
plt.axvline(4068,label='SII-4068,4076',ls='dashed',c='k')
plt.axvline(4076,ls='dashed',c='k')
plt.xlim((3900,4300))  # lim from renbin's paper fig 10
plt.ylim((-0.10,0.05))
plt.title('Bin {1}.{0} - Control Bin {1}.{0}'.format(check_bin,bin_num[2]))
plt.legend()
plt.show()


#%%
# How do I mask out emmission line regions?
lineless = ((strong-control)>-0.025)&((strong-control)<0.025)
plt.figure()
plt.plot(wave,strong/control)
plt.plot(wave[lineless],strong[lineless]/control[lineless])
plt.show()

fit = np.polyfit(wave[lineless],strong[lineless]/control[lineless],3)
fit_line = fit[0]*wave**3 + fit[1]*wave**2 + fit[2]*wave**1 + fit[3]*wave**0

# calc diff between fit and data[lineless], calculate stddev and cut gt 3 sigma
plt.figure()
plt.plot(wave,strong/control)
plt.plot(wave,fit_line)
plt.ylim(0.9,1.1)
plt.show()

residual_new = strong-control*fit_line
diff = strong[lineless]/control[lineless] - residual_new[lineless]
bad = diff>(np.mean(diff)+1*np.std(diff))
lineless[np.where(lineless==True)[0][bad]]=False
#%%

plt.figure(figsize=(20,7))

plt.subplot(1,2,1)
plt.plot(wave,strong,label='Bin 1')
plt.plot(wave,control,label='Control Bin 1')
plt.xlim((wave[0],wave[-1]))
plt.title('Stacked spectra: Bin {} / Sub-bin {}'.format(bin_num[2],check_bin))
plt.legend()
plt.show()

plt.subplot(1,2,2)
plt.plot(wave,residual_new)
plt.xlim((wave[0],wave[-1]))
plt.title('Bin {1}.{0} - Control Bin {1}.{0}'.format(check_bin,bin_num[2]))
#plt.ylim(-0.7,0.3)
plt.show()
#%%
lines = np.array([3728.6,3836.485,3869.86,3889.750,3890.158,3933, 3971.202,    # copy-pasted from Renbin's code
                  4070,4078,4102.899,4305,4341.692,4862.691,4959,5007,5200.317,
                  5756.19,5877.249, 6300, 6549.86, 6564.632,6585.27, 6718.294,6732.674])
lineless_new = np.full(len(wave),True)
width = np.full(len(wave),12)
plt.figure()
for i in range(len(lines)):
    lineless_new[((wave>(lines[i]-width[i]))&(wave<(lines[i]+width[i])))]=False
    plt.axvline(lines[i],c='k',alpha=0.7,ls='dashed')
plt.plot(wave,strong/control)
plt.plot(wave[lineless_new],strong[lineless_new]/control[lineless_new])
plt.show()
    