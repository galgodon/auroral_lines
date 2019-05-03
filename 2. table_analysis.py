# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 09:46:24 2019

@author: gerom
"""

import numpy as np
from astropy.io import fits
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
t = time.clock()
#import warnings                    # I put this here so the invalid value in sqrt warning would be be ignored
#warnings.filterwarnings("ignore")

from matplotlib import rc
font = { 'size' : 10 , 'family' : 'serif'}  # size of normal font
fs = 14  # size of titles, must be manually inserted using ',fontsize=fs'
rc('font', **font)

# =============================================================================
# Import table made in table.py
# =============================================================================

spaxel_data_table = fits.open('data/spaxel_data_table.fits')

def get_data(name,mask_name):
    return np.ma.MaskedArray(spaxel_data_table[1].data[name],mask=spaxel_data_table[1].data[mask_name]>0)

#%%
# =============================================================================
# 2a Find Quiescent Red spaxels using g-r vs D4000
# =============================================================================

### Pull Data for g-r and D4000
gmr = get_data('gmr','gmr_mask') - get_data('gmr_corr','gmr_mask')
D4000 = get_data('spec_index_D4000','spec_index_mask_D4000')

print('We start with {} spaxels'.format(len(D4000)))

### combine the masks
mask1 = np.invert(D4000.mask)&np.invert(gmr.mask)
print('We have {} spaxels after using D4000 and g-r masks'.format(mask1.sum()))

### Made by eye
xlim_qui = np.array([0.4,1.2])
ylim_qui = np.array([1.2,3.2])

### Make the cut
x_qui = np.linspace(xlim_qui[0],xlim_qui[1],len(gmr))   # this array is made to make the line look nicer
y_qui_1 = 1.6*x_qui + 0.62
y_qui_2 = 1.6*x_qui + 1.01

### Plot the data
plt.figure(figsize=(7,7))
plt.hist2d(gmr[mask1],D4000[mask1], bins=100, cmap=plt.cm.gray_r,
           norm=mpl.colors.LogNorm(),range=[xlim_qui,ylim_qui])

### Plot cut lines
plt.plot(x_qui,y_qui_1,ls='dashed',c='r')   
plt.plot(x_qui,y_qui_2,ls='dashed',c='r')

### Plot Settings
plt.xlim(xlim_qui)
plt.ylim(ylim_qui)
plt.xlabel('g-r',fontsize=fs)
plt.ylabel(r'$D_n(4000)$',fontsize=fs)
plt.savefig('graphs/3_quiescent_spx.png')
plt.show()

# Cut between the red lines
qui_spax = (mask1)&(D4000>(1.6*gmr + 0.62))&(D4000<(1.6*gmr + 1.01))  # Need to match y_qui_1 and 2
print('We have {} spaxels after cutting between red lines'.format(qui_spax.sum()))

######## these are observed g-r so there is expected to be an offset so we are looking at a bluer band
######## use k-correction of g-r (difference observed g-r and abs g-r in drpall)
######## also extinction correction
######## 22.5-2.5*LOG10(NSA_ELPETRO_FLUX) (make sure to find g - r indx 3 - 4)then subtract nsa_elpetro_absmag(make sure to find g - r indx 3 - 4) to find corr
######## this will be the sum of k corr and extinction
######## Put this into the table
#### to be expected. Horizontal smearing from correction not being accounted for

#%%
# =============================================================================
# Step 2, make S/N (Signal to Noise) histogram
# =============================================================================

snr_spx = spaxel_data_table[1].data['snr_spx']  # I did not use get_data because there is no mask
snr_bin = spaxel_data_table[1].data['snr_bin']

bins = 100

plt.figure(figsize=(5,5))
plt.hist((snr_spx[qui_spax]),bins=bins) 
plt.xlabel('S/N per Spaxel',fontsize=fs)
plt.ylabel('Frequency',fontsize=fs)
plt.show()

plt.figure(figsize=(5,5))
plt.hist((snr_bin[qui_spax]),bins=bins)
plt.xlabel('S/N per bin',fontsize=fs)
plt.ylabel('Frequency',fontsize=fs)
plt.show()

plt.figure(figsize=(5,5))
plt.hist(np.log10(snr_spx[qui_spax]),bins=bins)  
plt.xlabel('log S/N per Spaxel',fontsize=fs)
plt.ylabel('Frequency',fontsize=fs)
plt.show()

plt.figure(figsize=(5,5))
plt.hist(np.log10(snr_bin[qui_spax]),bins=bins)
plt.xlabel('log S/N per bin',fontsize=fs)
plt.ylabel('Frequency',fontsize=fs)
plt.show()

snrcut = 15
cut1 = (qui_spax)&(snr_spx>snrcut)
print('We have {} spaxels after cutting S/N < {}'.format(cut1.sum(),snrcut))

#%%
# =============================================================================
# Plot EWs against each other
# =============================================================================

### Pull out the data for EWs
ha_ew = get_data('summed_EW_H_alpha','summed_EW_mask_H_alpha')
ha_ew_ivar = get_data('summed_EW_IVAR_H_alpha','summed_EW_mask_H_alpha')

oII_ew = get_data('summed_EW_OII-3727','summed_EW_mask_OII-3727')
oII_ew_ivar = get_data('summed_EW_IVAR_OII-3727','summed_EW_mask_OII-3727')

nII_ew = get_data('summed_EW_NII-6585','summed_EW_mask_NII-6585')
nII_ew_ivar = get_data('summed_EW_IVAR_NII-6585','summed_EW_mask_NII-6585')

sII_ew_1 = get_data('summed_EW_SII-6718','summed_EW_mask_SII-6718')
sII_ew_2 = get_data('summed_EW_SII-6732','summed_EW_mask_SII-6732')
sII_ew = sII_ew_1 + sII_ew_2
sII_ew_1_ivar = get_data('summed_EW_IVAR_SII-6718','summed_EW_mask_SII-6718')
sII_ew_2_ivar = get_data('summed_EW_IVAR_SII-6732','summed_EW_mask_SII-6732')
sII_ew_ivar = 1/((1/sII_ew_1_ivar) + (1/sII_ew_2_ivar))

oIII_ew = get_data('summed_EW_OIII-5008','summed_EW_mask_OIII-5008')
oIII_ew_ivar = get_data('summed_EW_IVAR_OIII-5008','summed_EW_mask_OIII-5008')

hb_ew = get_data('summed_EW_H_beta','summed_EW_mask_H_beta')
hb_ew_ivar = get_data('summed_EW_IVAR_H_beta','summed_EW_mask_H_beta')

oI_ew = get_data('summed_EW_OI-6302','summed_EW_mask_OI-6302')
oI_ew_ivar = get_data('summed_EW_IVAR_OI-6302 ','summed_EW_mask_OI-6302')                   

### Plot them against each other
def plot_EW(EW,xlim,ylim,title):  # function that graphs EW vs H_alpha_EW using hist2d and log scale
    plt.figure(figsize=(5,5))
    hist_info = plt.hist2d(ha_ew[cut1],EW[cut1], bins=100, cmap=plt.cm.gray_r,
               norm=mpl.colors.LogNorm(), range=[xlim,ylim])  # PLOTS HALPHA AS X AXIS FOR ALL THE GRAPHS
    plt.plot(0,0,'x',c='r',label='(0,0)')  # Creates red x at origin
    plt.legend() 
    plt.xlabel(r'H$\alpha$ EW $[\AA]$',fontsize=fs)
    plt.ylabel(r'{} EW $[\AA]$'.format(title),fontsize=fs)
#    plt.title(r'H$\alpha$ vs {}'.format(title),fontsize=fs)
    return hist_info
 
p1 = plot_EW(oII_ew,[-2,10],[-5,40],r'OII $\lambda3727$')
p2 = plot_EW(nII_ew,[-2,10],[-2,5],r'NII $\lambda6549$')
p3 = plot_EW(sII_ew,[-2,10],[-3,10],r'SII $\lambda\lambda6718,6732$')
p4 = plot_EW(oIII_ew,[-2,10],[-1.5,6],r'OIII $\lambda5008$')
p5 = plot_EW(hb_ew,[-2,10],[-1.5,4],r'H$\beta$')
p6 = plot_EW(oI_ew,[-2,10],[-2,4],r'OI $\lambda6302$')
plt.show()

#%%  Use this to plot in a grid
#fig = plt.figure(figsize=(20,10))
#
#def plot_log_hist_2d_grid(j,a,xlim,ylim,title):  # program to make the grid
#    row=2  # change this to the desired number of rows in final graph
#    col=3  # same but columns
#    plt.subplot(row,col,j)  # j = position in grd, j=1 - Top Left; j=6 - Bottom right
#    hist_info = plt.hist2d(ha_ew[cut1],a[cut1], bins=100, cmap=plt.cm.gray_r,
#               norm=mpl.colors.LogNorm(), range=[xlim,ylim])  # PLOTS HALPHA AS X AXIS FOR ALL THE GRAPHS
#    plt.plot(0,0,'x',c='r',label='(0,0)')  # Creates red x at origin
#    plt.legend() 
#    plt.xlabel(r'H$\alpha$ EW $[\AA]$',fontsize=fs)
#    plt.ylabel(r'{} EW $[\AA]$'.format(title),fontsize=fs)
##    plt.title(r'H$\alpha$ vs {}'.format(title),fontsize=fs)
#    return hist_info
#
#p1 = plot_log_hist_2d_grid(1,oII_ew,[-2,10],[-5,40],r'OII $\lambda3727$')
#p2 = plot_log_hist_2d_grid(2,nII_ew,[-2,10],[-2,5],r'NII $\lambda6549$')
#p3 = plot_log_hist_2d_grid(3,sII_ew,[-2,10],[-3,10],r'SII $\lambda\lambda6718,6732$')
#p4 = plot_log_hist_2d_grid(4,oIII_ew,[-2,10],[-1.5,6],r'OIII $\lambda5008$')
#p5 = plot_log_hist_2d_grid(5,hb_ew,[-2,10],[-1.5,4],r'H$\beta$')
#p6 = plot_log_hist_2d_grid(6,oI_ew,[-2,10],[-2,4],r'OI $\lambda6302$')
#plt.savefig('graphs/4_EW.png')
#plt.show()
#
#def check_highest_bin(p):
#    print('There are {} counts in a bin of width {} whose bottom left corner is ({},{})'
#          .format(p[0][p[0]==np.max(p[0])][0],   p[1][1] - p[1][0],
#                  p[1][np.where(p[0]==np.max(p[0]))[0][0]],
#                  p[1][np.where(p[0]==np.max(p[0]))[0][0]]))
#
#check_highest_bin(p1)
#check_highest_bin(p2)
#check_highest_bin(p3)
#check_highest_bin(p4)
#check_highest_bin(p5)
#check_highest_bin(p6)

#%%
# =============================================================================
#  Exclude those with low [OII]/Halpha and those with high [OIII]/[OII].
#  EW([Oii])>5EW(Hα)−7  From Renbin's paper -- from Yan et al. (2006)
# =============================================================================

### Exclude low values by keeping the high values
high_oII_ha = oII_ew>(5*(ha_ew)-5)   ## changed after looking at plots from 7 to 5
cut2 = (cut1)&(high_oII_ha)
print('We have {} spaxels after cutting low OII/Halpha'.format(cut2.sum()))

### Exclude those with high [OIII]/[OII].
low_oIII_oII = oIII_ew/oII_ew <= 1
cut3 = (cut2)&(low_oIII_oII)
print('We have {} spaxels after cutting high OIII/OII'.format(cut3.sum()))

#%%
# =============================================================================
#  Separate strong-line sample and weak-line sample (according to a total EW formula).
#  Total EW index=EW(Hα)+1.03EW([Nii])+5.0EW([Oii])+0.5EW([Oiii])+EW([Sii])
# =============================================================================

tot_ew_indx = ha_ew + 1.03*nII_ew + 5*oII_ew + 0.5*(oIII_ew+sII_ew)

strong_line = tot_ew_indx>np.percentile(tot_ew_indx[cut3],75)  # seperated by the 75th percintile as in Renbin's paper

#%%
# =============================================================================
# For the strong-line sample,
#    a. bin them according to flux of [NII]/Halpha vs. [NII]/[SII] to different metallicity bins
#    b. Within each metallicity bin, further bin them according to different gas-star 
#       velocity offsets. (Make a histogram of gas-star velocity offset.)
# =============================================================================
# =============================================================================
# Make the initial graph of [NII]/Halpha vs. [NII]/[SII] BEFORE binning
# =============================================================================
### Take out flux and IVAR data
flux_nII = get_data('gaus_flux_NII-6585','gaus_flux_mask_NII-6585')
flux_ivar_nII = get_data('gaus_flux_IVAR_NII-6585','gaus_flux_mask_NII-6585')

flux_ha = get_data('gaus_flux_H_alpha','gaus_flux_mask_H_alpha')
flux_ivar_ha = get_data('gaus_flux_IVAR_H_alpha','gaus_flux_mask_H_alpha')

flux_sII_1 = get_data('gaus_flux_SII-6718','gaus_flux_mask_SII-6718')
flux_sII_2 = get_data('gaus_flux_SII-6732','gaus_flux_mask_SII-6732')
flux_sII = flux_sII_1 + flux_sII_2
flux_ivar_sII_1 = get_data('gaus_flux_IVAR_SII-6718','gaus_flux_mask_SII-6718')
flux_ivar_sII_2 = get_data('gaus_flux_IVAR_SII-6732','gaus_flux_mask_SII-6732')
flux_ivar_sII = 1/((1/flux_ivar_sII_1) + (1/flux_ivar_sII_2))

### Take the log of the ratios
lr_nII_ha = np.ma.log10(flux_nII)-np.ma.log10(flux_ha)    # lr = line ratio
lr_nII_sII = np.ma.log10(flux_nII)-np.ma.log10(flux_sII)   
lr_mask = (np.invert(lr_nII_ha.mask))&(np.invert(lr_nII_sII.mask))  # Seperate non-masked values

### fractional error of ratio
# fractional error adds quadratically
# (sigx/x)^2 + (sigy/y)^2   then sqrt
# change log to 10 base log 
# div by ln(10)
frac_err_nII_ha = np.ma.sqrt(1/((flux_nII)**2*(flux_ivar_nII)) + 1/((flux_ha)**2*(flux_ivar_ha))) / np.log(10)
frac_err_nII_oII = np.ma.sqrt(1/((flux_nII)**2*(flux_ivar_nII)) + 1/((flux_sII)**2*(flux_ivar_sII))) / np.log(10)

#### Check frac_err cut
#### Therefore, we require that the frac-tional errors on [Nii]/[Oii] and [Nii]/Hαratios to be betterthan 0.3 dex
#plt.hist(frac_err_nII_ha[(cut3)&(not_nan)&(strong_line)],range=(0,1))
#plt.hist(frac_err_nII_oII[(cut3)&(not_nan)&(strong_line)],range=(0,1))

cut4 = (cut3)&(lr_mask)&(strong_line)&(frac_err_nII_ha<0.3)&(frac_err_nII_oII<0.3)
print('We have {} spaxels with strong lines and good frac errors'.format(cut4.sum()))

plt.figure(figsize=(5,5))
xlim_lr = np.array([-1,1])
ylim_lr = np.array([-1,1])
### plot 2d hist of the line ratios
plt.hist2d(lr_nII_ha[cut4],lr_nII_sII[cut4], bins=100, cmap=plt.cm.gray_r,
           norm=mpl.colors.LogNorm(),range=[xlim_lr,ylim_lr])

### Plot cut lines   ### plot through the tails
x_lr = np.linspace(xlim_lr[0],xlim_lr[1],100)
y_lr = 1.2*x_lr - 0.05
plt.plot(x_lr,y_lr,ls='dashed',c='r')

plt.xlabel(r'log [NII]/H$\alpha$',fontsize=fs)
plt.ylabel('log [NII]/[SII]',fontsize=fs)
plt.xlim(xlim_lr)
plt.ylim(ylim_lr)
plt.show()

#%%
# =============================================================================
# Rotate the graph so my 'fit line' is horizontal, this makes it easy to split into thirds using the percentile function
# =============================================================================
### Rotate the graph in order to cut into thirds
def rotate(theta,x,y):
    c,s = np.cos(theta),np.sin(theta)
    x_rot = c*x - s*y
    y_rot = s*x + c*y
    return x_rot,y_rot

### Rotate data
theta = -np.arctan(1.2)  # slope of y_lr  since m = arctan(theta), we rotate by -arctan(theta)
x_rot,y_rot = rotate(theta,lr_nII_ha,lr_nII_sII)
x_lr_rot,y_lr_rot = rotate(theta,x_lr,y_lr)
xlim_lr_rot = np.array([-1,1])
ylim_lr_rot = np.array([-1,1])

### Plot rotated data
plt.figure(figsize=(5,5))
plt.hist2d(x_rot[cut4],y_rot[cut4], bins=100, cmap=plt.cm.gray_r,
           norm=mpl.colors.LogNorm(),range=[xlim_lr_rot,ylim_lr_rot])
plt.plot(x_lr_rot,y_lr_rot,ls='dashed',c='r')

### cut the data into thirds
x_lr_33 = np.full(100,np.percentile(x_rot[cut4],33.33))
x_lr_66 = np.full(100,np.percentile(x_rot[cut4],66.66))
y_lr_33 = np.linspace(ylim_lr_rot[0],ylim_lr_rot[1],100)
plt.plot(x_lr_33,y_lr_33,ls='dashed',c='b')
plt.plot(x_lr_66,y_lr_33,ls='dashed',c='b')


plt.xlabel(r'$x_{rot}$',fontsize=fs)
plt.ylabel(r'$y_{rot}$',fontsize=fs)
plt.xlim(xlim_lr_rot)
plt.ylim(ylim_lr_rot)
plt.show()

#%%
# =============================================================================
# Re-make the graph with the cutlines made in the above cell
# =============================================================================
### Rotate the cut lines
x_bin_cut1,y_bin_cut1 = rotate(-theta,x_lr_33,y_lr_33)
x_bin_cut2,y_bin_cut2 = rotate(-theta,x_lr_66,y_lr_33)

### Remake the plot with new cut lines
plt.figure(figsize=(7,7))
plt.hist2d(lr_nII_ha[cut4],lr_nII_sII[cut4], bins=100, cmap=plt.cm.gray_r,  # a copy-paste from above
           norm=mpl.colors.LogNorm(),range=[xlim_lr,ylim_lr])
plt.plot(x_lr,y_lr,ls='dashed',c='r')

plt.plot(x_bin_cut1,y_bin_cut1,ls='dashed',c='b')  # New part
plt.plot(x_bin_cut2,y_bin_cut2,ls='dashed',c='b')

plt.xlabel(r'log [NII]/H$\alpha$',fontsize=fs)
plt.ylabel('log [NII]/[SII]',fontsize=fs)
plt.xlim(xlim_lr)
plt.ylim(ylim_lr)

plt.text(-0.45,0.75,r'High [NII]/H$\alpha$',fontsize=fs)
plt.text(-0.70,0.50,r'Mid [NII]/H$\alpha$',fontsize=fs)
plt.text(-0.85,0.15,r'Low [NII]/H$\alpha$',fontsize=fs)
#plt.savefig('graphs/5_metallicity_bins.png')
plt.show()

### Create metalicity bins
bin1_high_nII_ha = (cut4)&(x_rot>np.percentile(x_rot[cut4],66.66))
bin2_mid_nII_ha = (cut4)&(x_rot>np.percentile(x_rot[cut4],33.33))&(x_rot<np.percentile(x_rot[cut4],66.66))
bin3_low_nII_ha = (cut4)&(x_rot<np.percentile(x_rot[cut4],33.33))

#%%
# =============================================================================
# b. Within each metallicity bin, further bin them according to different gas-star 
#    velocity offsets. (Make a histogram of gas-star velocity offset.)
# =============================================================================
### Pull data to create a histogram of gas-star velocity offsets
vel_offset = get_data('stell_vel','stell_vel_mask') - get_data('ha_vel','ha_vel_mask')

plt.figure(figsize=(20,5))

plt.subplot(1,3,1)
bin1_hist = plt.hist(vel_offset[bin1_high_nII_ha],range=(-500,500),bins=25)
plt.title(r'High [NII]/H$\alpha$')
plt.xlabel('Velocity offset (Stellar - Gas)',fontsize=fs)
#plt.show()

plt.subplot(1,3,2)
bin2_hist = plt.hist(vel_offset[bin2_mid_nII_ha],range=(-500,500),bins=25)
plt.title(r'Mid [NII]/H$\alpha$')
plt.xlabel('Velocity offset (Stellar - Gas)',fontsize=fs)
#plt.show()

plt.subplot(1,3,3)
bin3_hist = plt.hist(vel_offset[bin3_low_nII_ha],range=(-500,500),bins=25)
plt.title(r'Low [NII]/H$\alpha$')
plt.xlabel('Velocity offset (Stellar - Gas)',fontsize=fs)

plt.savefig('graphs/6_vel_off.png')
plt.show()

def split_bins(bin_num,bin_hist):
    # Creates bin_splits based on the histograms above. Change the histogram, change the bin_split
    bin_split = np.zeros( ( len(bin_hist[0]) , len(vel_offset[bin_num]) ) ,dtype='bool')
    bin_avg = np.zeros(len(bin_hist[0]))
    for i in range(len(bin_hist[1]) - 1):
        bin_split[i] = (vel_offset[bin_num]>bin_hist[1][i]) & (vel_offset[bin_num]<bin_hist[1][i+1])
        bin_avg[i] = np.ma.average(vel_offset[bin_num][bin_split[i]])
    return bin_split, bin_avg
        
bin1_split,bin1_avg = split_bins(bin1_high_nII_ha,bin1_hist)
bin2_split,bin2_avg = split_bins(bin2_mid_nII_ha,bin2_hist)
bin3_split,bin3_avg = split_bins(bin3_low_nII_ha,bin3_hist)

#np.sum(bin1_split,axis=1)

#%%
# =============================================================================
# 6. Build control sample for each bin. (matching them in 3D space of Vdisp, D4000, and flux in r-band)
# =============================================================================

# in the multi-dimensional space with each of the semi-axis equal to two times the 
# median uncertainty in EW foreach line

med_ew_ivar_ha = np.ma.median(ha_ew_ivar[cut1])   
med_ew_ivar_oII = np.ma.median(oII_ew_ivar[cut1]) 
med_ew_ivar_nII = np.ma.median(nII_ew_ivar[cut1])
med_ew_ivar_sII = np.ma.median(sII_ew_ivar[cut1])
med_ew_ivar_oIII = np.ma.median(oIII_ew_ivar[cut1])
med_ew_ivar_hb = np.ma.median(hb_ew_ivar[cut1])

mult_eli = ((ha_ew)**2*med_ew_ivar_ha + (oII_ew)**2*med_ew_ivar_oII + 
            (nII_ew)**2*med_ew_ivar_nII + (sII_ew)**2*med_ew_ivar_sII + 
            (oIII_ew)**2*med_ew_ivar_oIII + (hb_ew)**2*med_ew_ivar_hb < 36 )
zero_cut = (mult_eli)&(cut1)&(np.invert(strong_line))
print('Zero line sample has {} spaxels'.format(zero_cut.sum()))


#%%
# =============================================================================
# Create all the functions needed to create the control sample
# =============================================================================
from scipy import spatial  # import this to use KDTree and to query the KDTree
import astropy.constants   # this is just used to get the speed of light c
c = astropy.constants.c.to('km/s').value

vdisp = get_data('stell_sigma_cor','stell_sigma_mask')                 # get dispersion velocity
flux_r_band = np.log10((spaxel_data_table[1].data['flux_r_band']*10**  # get flux_r_band using the formula: log( RIMG*10^(-0.4*EBVGAL*2.75) * (1+z/c)^4 )
               (-0.4*spaxel_data_table[1].data['gal_red_B-V']*2.75))*
               (1+spaxel_data_table[1].data['z_vel']/c)**4)

def redo_gt_2(used_indx):
    # Function that checks which spaxels to redo in the search for a control sample
    # Here the condition is to redo a spaxel if it appears more than TWICE in the
    # control group. 
    # array = array of the number indices of the spaxels to be used in the control sample
    # EX: array = ([[3,7],[8,2],[2,5],[2,3]]) the spaxel with index '2' is used 3 times
    # so the output will be: ([False, False, False, False, False, False,  True, False])
    # notice this code flattens the array. Since things can be repeated twice, only the 
    # third instance of 2 is marked to be re-done
    unique, return_inverse, counts = np.unique(used_indx,return_inverse=True,return_counts=True)
    
    redo = np.full(len(return_inverse),False)    # initialize array with False: Meaning don't redo anything
    for i in range(len(np.where(counts>2)[0])):
        count_gt_2 = np.where(return_inverse==np.where(counts>2)[0][i])[0][2:]  # check where there are more than two counts
        redo[count_gt_2] = True  
    return redo

def check_unused_indx(array,used_indx):
    # array = only the length of this matters, it must be the len(array)==# of spaxels in total control sample
    # used_indx = array of the number indices of the spaxels to be used in this specific control sample
    # EX: if there are 10 spaxels in the total control sample (that is len(array)=10)and the specific control sample only uses 
    #     the 3rd and 5th spaxel (that is used_indx=[2,4]), this will return 
    #     array([ True,  True, False,  True,  True, False,  True,  True,  True, True])
    unused=np.full(len(array),False)  # True if indx does not appear in used_indx
    for i in range(len(array)):
        if (i==used_indx).sum()==0:
            unused[i] = True
    return unused

def norm_0_1(array,indx):
    # this shifts the array so that the 5th and 95th percentile of array[indx] are 0 and 1 respectively
    # note: this will shift the entire array but the shift is only based on the indexed array
    return ((array-np.percentile(array[indx],5)) / 
            (np.percentile(array[indx],95)-np.percentile(array[indx],5)))
    
norm_vdisp = norm_0_1(vdisp,zero_cut)  # normalize based on vdisp[zero_cut] but output the entire shifted array
norm_D4000 = norm_0_1(D4000,zero_cut)  # same
norm_flux_r_band = norm_0_1(flux_r_band,zero_cut)

control = spatial.KDTree(list(zip(norm_vdisp[zero_cut],norm_D4000[zero_cut],   # create a KDTree based on the normalized array
                                  norm_flux_r_band[zero_cut])))

def get_control(bin_cut,bin_split,i):    
    dist, bin_control = control.query(list(zip(norm_vdisp[bin_cut][bin_split],    # do initial search for nearest 2 neighbors (k=2)
                                                norm_D4000[bin_cut][bin_split],   # 
                                                norm_flux_r_band[bin_cut][bin_split])),k=2)
    dist,bin_control = dist.ravel(),bin_control.ravel()
    redo = redo_gt_2(bin_control)  # see if we have to redo anything
#    if redo.sum()!=0:
    print('We have to redo {}/{} queries in bin {}'.format(redo.sum(),len(redo),i))
    max_itt = 10
    itt = 0
    while (redo.sum() != 0):   # loops until redo.sum() is 0
        # check what indices were not used
        unused = check_unused_indx(norm_vdisp[zero_cut],bin_control) 
        # make new control sample out of unused indices
        control_new = spatial.KDTree(list(zip(norm_vdisp[zero_cut][unused],  
                                          norm_D4000[zero_cut][unused],norm_flux_r_band[zero_cut][unused])))
        # Search again for each spaxel with the new control
        dist_new, bin_control_new = control_new.query(list(zip(norm_vdisp[bin_cut][bin_split],  
                                                norm_D4000[bin_cut][bin_split],
                                                norm_flux_r_band[bin_cut][bin_split])),k=2)   
        # Turn the new index into old index so we can replace them
        bin_control_new = np.where(unused==True)[0][bin_control_new]  
        # Replace only the ones we need to redo
        bin_control[redo]=bin_control_new.ravel()[redo]
        dist[redo] = dist_new.ravel()[redo]
        redo = redo_gt_2(bin_control)  # see if we have to redo anything still
#        print('We have to redo {} queries'.format(redo.sum()))
        itt += 1 
        if itt == max_itt:   # stops infinite loop as long as max_itt is an int > 1
            print('Reached max itteration of {}'.format(max_itt))
            break
    bin_control = np.where(zero_cut.data==True)[0][bin_control]   # changes indices to global indx only works if control is based on zero_cut
    return bin_control,dist

def get_all_control(bin_cut,bin_split):
    control_all = []
    dist_all = []
    for i in range(len(bin_split)):
        if np.sum(bin_split[i])==0:
            control_all.append(np.array([]))
            dist_all.append(np.array([]))
        else:
            control, dist = get_control(bin_cut,bin_split[i],i+1)
            control_all.append(control)
            dist_all.append(dist)
    return control_all,dist_all

#%%
# =============================================================================
# Create the control sample
# =============================================================================
bin1_control,bin1_dist = get_all_control(bin1_high_nII_ha,bin1_split)   # 
print()
bin2_control,bin2_dist = get_all_control(bin2_mid_nII_ha,bin2_split)
print()
bin3_control,bin3_dist = get_all_control(bin3_low_nII_ha,bin3_split)
print()


#a = np.unique(bin1_control[12],return_counts=True)
#aa = plt.hist(a[1],bins=np.arange(10))
#plt.show()
#%%
# =============================================================================
# Limit the max_distance between neighboring points and remake bin_split arrays
# also remake bin_control
# =============================================================================
max_dist = np.sqrt(3*0.05**2) # 3 dimensions 95th-5th = 1 when norm , 1/10=0.1, 0.1/2=0.05

def remake_bin_split(dist,indx,bin_split,bin_num):   # remake bin_split arrays but exlude values where nearest point is < max_dist
    bin_split_new = []                               # this also makes the two new arrays a global indx (see return)
    for i in range(len(dist)):
        if len(dist[i])!=0:
            where = np.where(dist[i]<max_dist)[0]
            for j in range(len(where)): # make indx inside where reresent indx inside strong_line bin
                if where[j]%2==0:   # if the index is even
                    where[j]=where[j]/2
                else:  # if the index is odd
                    where[j]=(where[j]-1)/2
            unique,counts = np.unique(where,return_counts=True)
            good = unique[counts==2]
            new_split = np.where(bin_num.data==True)[0][np.where(bin_split[i]==True)[0]][good]
            bin_split_new.append(new_split)
        else:
            bin_split_new.append(np.array([]))
            
    bin_control_new = indx.copy()
    for i in range(len(bin_control_new)):
        gt_max = np.where(dist[i]>=max_dist)[0]
        if len(gt_max)!=0:
            delete = np.array([],dtype='i4')
            for j in range(len(gt_max)):
                if gt_max[j]%2==0:
                    delete = np.append(delete,[gt_max[j],gt_max[j]+1])
                else:
                    delete = np.append(delete,[gt_max[j],gt_max[j]-1])
            bin_control_new[i] = np.delete(bin_control_new[i],delete)
    return bin_split_new,bin_control_new
                
bin1_split_new,bin1_control_new = remake_bin_split(bin1_dist,bin1_control,bin1_split,bin1_high_nII_ha)
bin2_split_new,bin2_control_new = remake_bin_split(bin2_dist,bin2_control,bin2_split,bin2_mid_nII_ha)
bin3_split_new,bin3_control_new = remake_bin_split(bin3_dist,bin3_control,bin3_split,bin3_low_nII_ha)

#%%
def save_bin(bin_split,bin_avg,bin_control,num):
    bin_s = fits.HDUList()
    for i in range(len(bin_split)):
        bin_s.append(fits.ImageHDU(bin_split[i],name='{}_{}'.format(num,i+1)))
    bin_s.append(fits.ImageHDU(bin_avg,name='AVG_OFFSET_SUBBIN'))
    bin_s.writeto('data/Bin_{}.fits'.format(num),overwrite=True)

    bin_c = fits.HDUList()
    for i in range(len(bin_control)):
        bin_c.append(fits.ImageHDU(bin_control[i],name='{}_{}'.format(num,i+1)))
    bin_c.append(fits.ImageHDU(bin_avg,name='AVG_OFFSET_SUBBIN'))
    bin_c.writeto('data/Bin_{}_Control.fits'.format(num),overwrite=True)
    
#save_bin(bin1_split_new,bin1_avg,bin1_control_new,1)
#save_bin(bin2_split_new,bin2_avg,bin2_control_new,2)
#save_bin(bin3_split_new,bin3_avg,bin3_control_new,3)

#%% END
print('Code execution Time: {} sec'.format(time.clock()-t))