# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 13:23:10 2019

@author: Gerome Algodon

Finding quiescent red sequence galaxies in MaNGA.
Recreating the first two plots in Renbin's Paper
https://ui.adsabs.harvard.edu/#abs/2018MNRAS.481..476Y/abstract
"""
#%%
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.table import Table
import matplotlib as mpl

from matplotlib import rc
font = { 'size' : 10 , 'family' : 'serif'}  # size of normal font
fs = 14  # size of titles, must be manually inserted using ',fontsize=fs'
rc('font', **font)

# =============================================================================
# Inspired by: https://python-graph-gallery.com/86-avoid-overlapping-in-scatterplot-with-2d-density/
# Code to make the shaded density graph
# =============================================================================
#%%
from scipy.stats import kde

def shaded_density_graph(x,y,xlim,ylim,nbins=40):
    # needed imports: scipy.stats.kde, numpy, matplotlib.pyplot
    # inputs: scatterplot data (x,y) ; xlim,ylim need to be len 2 arrays
    # Very similar to 2D historgram, nbins is how many bins to split data into even bins
    indx = (x>xlim[0])&(x<xlim[1])&(y>ylim[0])&(y<ylim[1])
    x = x[indx]
    y = y[indx]
    k = kde.gaussian_kde((x,y))
    xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.gray_r)
#%%
# =============================================================================
# Import dapall and drpall files
# =============================================================================

dap_all = fits.open('data/dapall-v2_4_3-2.2.1.fits')
drp_all = fits.open('data/drpall-v2_4_3.fits')
#%%
# =============================================================================
# Pull out plateifu from both dap and drp. From dap pull out D4000, 
# from drp pull out g and r band 
# =============================================================================

dapindx = (dap_all[1].data['DAPTYPE']=='HYB10-GAU-MILESHC')&(dap_all[1].data['DAPDONE']==1)
plate_ifu_dap = dap_all[1].data['plateifu'][dapindx]
D4000 = dap_all[1].data['SPECINDEX_1RE'][:,43][dapindx]  # indx 43 just like in table.py

drpindx = dap_all[1].data['DRPALLINDX'][dapindx]
plate_ifu_drp = drp_all[1].data['plateifu'][drpindx]
g = drp_all[1].data['nsa_elpetro_absmag'][drpindx,3]  # fnugriz
r = drp_all[1].data['nsa_elpetro_absmag'][drpindx,4]
g_r = g - r

#%%
# =============================================================================
# Plot r vs g-r
# =============================================================================

xlim = np.array([-24,-15])   # created by eye
ylim = np.array([0.1,1])

plt.figure(figsize=(5,5))
plt.plot(r,g_r,'.',c='k',alpha=0.1)  
plt.xlim(xlim)
plt.ylim(ylim)

plt.xlabel(r'$M_r$',fontsize=fs)
plt.ylabel('g-r',fontsize=fs)
#plt.title('r vs g-r')
plt.show()

#%%
# =============================================================================
# Empirically create cut of red sequence galaxies and overplot it
# =============================================================================

x = np.linspace(xlim[0],xlim[1],len(r))   # this array is made to make the line look nicer

y1 = -0.015*x + 0.39   # lower line, this is used to graph the line
y2 = -0.015*x + 0.47   # upper line

y1r = -0.02*x + 0.49   # lower line, this is used to graph the line
y2r = -0.02*x + 0.59   # upper line

y11 = -0.015*r + 0.39  # this is a copy paste of y1 and y2, this is made to make the cut accurate
y22 = -0.015*r + 0.47

#plt.plot(r,g_r,'.',c='k',alpha=0.1)  # re-make r vs g-r
plt.figure(figsize=(7,7))
#shaded_density_graph(r,g_r,xlim,ylim)   # replacing shaded density graph with 2d historgram
plt.hist2d(r,g_r, bins=50, cmap=plt.cm.gray_r,norm=mpl.colors.LogNorm(), range=[xlim,ylim])
plt.xlim(xlim)
plt.ylim(ylim)

plt.plot(x,y1,ls='dashed',c='r')   # plot the cut lines
plt.plot(x,y2,ls='dashed',c='r')

#plt.plot(x,y1r,ls='dashed',c='b')   # plot the cut lines
#plt.plot(x,y2r,ls='dashed',c='b')

plt.xlabel(r'$M_r$',fontsize=fs)
plt.ylabel('g-r',fontsize=fs)
#plt.title('r vs g-r: Density Plot')
#plt.savefig('graphs/1_red_seq.png')
plt.show()

#%%
# =============================================================================
# Plot the cut we made to make sure that it did what we want it to do
# =============================================================================

red_seq = (g_r>=y11)&(g_r<=y22)  # get the area in between both lines

plt.figure()
plt.plot(r[red_seq],g_r[red_seq],'.',c='k',alpha=0.1)  # plot just that area
plt.xlim(xlim)
plt.ylim(ylim)

plt.xlabel('r')
plt.ylabel('g-r')
plt.title('r vs g-r: Red Sequence Cut')
plt.show()

#%%
# =============================================================================
# Now we need to plot D4000 vs g-r, but the problem is that the 
# plate-ifu don't line up, so this lines them up
# the first cut applies to the drp data
#    THIS IS NO LONGER NECESSARY
#       In the weekly meeting Kyle taught me how to make plate and IFU line up between the data sets
# =============================================================================

#D4000_red_seq = np.zeros(red_seq.sum())  # initialize new D4000 array
#
#
#for i in range(red_seq.sum()):  
#    D4000_red_seq[i] = D4000[plate_ifu_dap==plate_ifu_drp[red_seq][i]][0]  
    
# why is there two?
#i = 0
#print(D4000[plate_ifu_dap==plate_ifu_drp[red_seq][i]][0])
#print(D4000[plate_ifu_dap==plate_ifu_drp[red_seq][i]][1])
    
#for i in range(red_seq.sum()):
#    a = D4000[plate_ifu_dap==plate_ifu_drp[red_seq][i]][0]
#    b = D4000[plate_ifu_dap==plate_ifu_drp[red_seq][i]][1]
#    if not (a >= b-0.3)&(a <= b+0.3):
#        print('{} vs {}'.format(a,b))

#%%
# =============================================================================
# Graph g-r vs d4000
# =============================================================================

xlim2 = np.array([0.625,0.825])  # also made by eye
ylim2 = np.array([1.4,2.4])

plt.figure()
plt.plot(g_r[red_seq], D4000[red_seq], '.', c='k', alpha=0.1)
plt.xlim(xlim2)
plt.ylim(ylim2)

plt.xlabel('g-r')
plt.ylabel('D4000')
plt.title('g-r vs D4000')
plt.show()

#%%
# =============================================================================
#  Empirically create cut and overplot it over the data
# =============================================================================

x2 = np.linspace(xlim2[0],xlim2[1],red_seq.sum())  # this array is made to make the line look nicer

y3 = 1.6*x2 + 0.90    # lower line, this is used to graph the line
y4 = 1.6*x2 + 1.01   # upper line

y3r = 1.6*x2 + 0.26    # lower line, this is used to graph the line
y4r = 1.6*x2 + 0.52   # upper line

y33 = 1.6*g_r[red_seq] + 0.90   # this is a copy paste of y3 and y4, this is made to make the cut accurate
y44 = 1.6*g_r[red_seq] + 1.01

#plt.plot(g_r[red_seq], D4000[red_seq], '.', c='k', alpha=0.1)  # re-make g-r vs D4000
plt.figure()
#shaded_density_graph(g_r[red_seq], D4000[red_seq],xlim2,ylim2)
plt.hist2d(g_r[red_seq],D4000[red_seq], bins=30, cmap=plt.cm.gray_r,norm=mpl.colors.LogNorm(), range=[xlim2,ylim2])
plt.xlim(xlim2)
plt.ylim(ylim2)

plt.plot(x2,y3,ls='dashed',c='r')   # plot the cut lines
plt.plot(x2,y4,ls='dashed',c='r')

#plt.plot(x2,y3r,ls='dashed',c='b')   # plot the cut lines
#plt.plot(x2,y4r,ls='dashed',c='b')

plt.xlabel('g-r',fontsize=fs)
plt.ylabel(r'$D_n(4000)$',fontsize=fs)
#plt.title('g-r vs D4000: Density Plot')
#plt.savefig('graphs/2_quiescent.png')
plt.show()

#%%
# =============================================================================
# Plot the cut we made to make sure that it did what we want it to do
# quiescent red sequence galaxies
# =============================================================================

qui_red_seq = (D4000[red_seq]>=y33)&(D4000[red_seq]<=y44)  # get the area in between both lines

plt.figure()
plt.plot(g_r[red_seq][qui_red_seq], D4000[red_seq][qui_red_seq], '.', c='k', alpha=0.1)   # plot just that area
plt.xlim(xlim2)
plt.ylim(ylim2)

plt.xlabel('g-r')
plt.ylabel('D4000')
plt.title('g-r vs D4000: Quiescent red galaxy cut')
plt.show()

# =============================================================================
# save the plate and ifu of the final cut of  
# =============================================================================
    
plate = drp_all[1].data['plate'][drpindx][red_seq][qui_red_seq]
ifudsgn = drp_all[1].data['ifudsgn'][drpindx][red_seq][qui_red_seq]

quiescent_red_sequence_galaxies = Table([plate,ifudsgn],names=['plate','ifudsgn'])

#quiescent_red_sequence_galaxies.write('data/quiescent_red_sequence_galaxies.fits', format='fits',overwrite=True)
