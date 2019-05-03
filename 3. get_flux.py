# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 11:13:50 2019

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
import sys

spaxel_data_table = fits.open('data/spaxel_data_table.fits')

bin1,bin1_control = fits.open('data/Bin_1.fits'),fits.open('data/Bin_1_Control.fits')
bin2,bin2_control = fits.open('data/Bin_2.fits'),fits.open('data/Bin_2_Control.fits')
bin3,bin3_control = fits.open('data/Bin_3.fits'),fits.open('data/Bin_3_Control.fits')

#%%
# =============================================================================
# Create function that will find flux in each spaxel given the bin
# =============================================================================
def get_flux(bin_num,name):
    ### initialize the fits file and the lists that will fill the fits file
    flux_bin = fits.HDUList()     
    flux_array,ivar_array,mask_array,predisp_array = [],[],[],[]
    
    ### First find unique galaxies in the total bin so we don't have to pull the same galaxy twice in a bin
    ### Fill flux_array with empty files to be filled later then combine all subbins into totbin
    for t in range(len(bin_num)):
        flux_array.append(np.zeros( (len(bin_num[t].data) , 4563) ))
        ivar_array.append(np.zeros( (len(bin_num[t].data) , 4563) ))
        mask_array.append(np.zeros( (len(bin_num[t].data) , 4563) ))
        predisp_array.append(np.zeros( (len(bin_num[t].data) , 4563) ))
        if t==0:
            totbin = bin_num[t].data
            subbin = np.full(len(bin_num[t].data),t,dtype='int64')
            spxnum = np.arange(len(bin_num[t].data),dtype='int64')
        else:
            totbin = np.append(totbin,bin_num[t].data)
            subbin = np.append(subbin,np.full(len(bin_num[t].data),t,dtype='int64'))
            spxnum = np.append(spxnum,np.arange(len(bin_num[t].data),dtype='int64'))
    totbin,subbin,spxnum = totbin.astype(int),subbin.astype(int),spxnum.astype(int)
            
    ### Pull ID for every spaxel: plate-ifu = galaxy, spaxel_id = specific spaxel
    plate = spaxel_data_table[1].data['plate'][totbin]
    ifu = spaxel_data_table[1].data['ifu'][totbin]
    spaxel_id = spaxel_data_table[1].data['emline_ID'][totbin]
    
    ###Combine plate and ifu to find unique galaxies 
    plate_ifu = np.zeros((len(plate)),dtype='U25')
    for j in range(len(plate)):
        plate_ifu[j] = '{}-{}'.format(plate[j],ifu[j])
    plate_ifu_unique, u_indx = np.unique(plate_ifu,return_index=True)
    plate_unique, ifu_unique = plate[u_indx], ifu[u_indx]
    
    for i in range(len(plate_unique)):
        ### to visually show progress
        sys.stdout.write('\r'+('     Galaxy {}/{}'.format(i+1,len(plate_unique))))  
                              
        ### Load files
        maps_file = ('/data/manga/spectro/analysis/MPL-7/HYB10-GAU-MILESHC/{0}/{1}/manga-{0}-{1}-MAPS-HYB10-GAU-MILESHC.fits.gz'.format(plate_unique[i],ifu_unique[i]))
        maps = fits.open(maps_file)
        logcube_file = ('/data/manga/spectro/redux/MPL-7/{0}/stack/manga-{0}-{1}-LOGCUBE.fits.gz'.format(plate_unique[i],ifu_unique[i]))
        log = fits.open(logcube_file)
        
        ### make the primary the wavelength array
        if len(flux_bin)==0:  
            flux_bin.append(fits.PrimaryHDU(log['WAVE'].data))
        
        ### Find all the spaxels that are in this galaxy. The len of this should reflect the count output of the np.unique function used above
        spx_in_gal = np.where(plate_ifu_unique[i]==plate_ifu)[0]
        
        for s in range(len(spx_in_gal)): # for every spaxel in this unique galaxy pull out the 
             
            ### Find the coordinate for this spaxel using the maps file
            spaxel_coord = (spaxel_id[spx_in_gal][s]==maps['BINID'].data[3,:,:])
            assert spaxel_coord.sum()==1    # make sure there is only one match
            
            ### Pull out the needed arrays from the logcube file and place it in the f indx
            subbin_num = subbin[spx_in_gal[s]]  # find subbin number of this spaxel
            spxnum_num = spxnum[spx_in_gal[s]]  # find slot to put in this specific spaxel
            flux_array[subbin_num][spxnum_num,:] = log['FLUX'].data[:,spaxel_coord].flatten()
            ivar_array[subbin_num][spxnum_num,:] = log['IVAR'].data[:,spaxel_coord].flatten()
            mask_array[subbin_num][spxnum_num,:] = log['MASK'].data[:,spaxel_coord].flatten()
            predisp_array[subbin_num][spxnum_num,:] = log['PREDISP'].data[:,spaxel_coord].flatten()
            
    print()
    for b in range(len(flux_array)):
        sys.stdout.write('\r'+('     Sub-bin {}/{}'.format(b+1,len(flux_array)))) 
        flux_bin.append(fits.ImageHDU(flux_array[b],name='FLUX_SUBBIN_{}'.format(b+1)))
        flux_bin.append(fits.ImageHDU(ivar_array[b],name='IVAR_SUBBIN_{}'.format(b+1)))
        flux_bin.append(fits.ImageHDU(mask_array[b],name='MASK_SUBBIN_{}'.format(b+1)))
        flux_bin.append(fits.ImageHDU(predisp_array[b],name='PREDISP_SUBBIN_{}'.format(b+1)))
    ### save as fits file once we went through all of the subbins
    flux_bin.writeto('data/{}.fits'.format(name),overwrite=True)
        
print('Working on Bin 1')
get_flux(bin1,'flux_bin_1')
print()
print('Working on Bin 2')
get_flux(bin2,'flux_bin_2')
print()
print('Working on Bin 3')
get_flux(bin3,'flux_bin_3')
print()
print('Working on Bin 1 Control')
get_flux(bin1_control,'flux_bin_1_control')
print()
print('Working on Bin 2 Control')
get_flux(bin2_control,'flux_bin_2_control')
print()
print('Working on Bin 3 Control')
get_flux(bin3_control,'flux_bin_3_control')     
        
#%% END
print('Code execution Time: {} sec'.format(time.clock()-t)) 