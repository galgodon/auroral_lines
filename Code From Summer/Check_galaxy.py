# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 19:30:25 2018

@author: Gerome Algodon
"""

import matplotlib.pyplot as plt
from astropy.io import fits
import os

plt.ioff()  # Turn off plotting

log = fits.open('data/MPL-6_master_catalogue_Apr9.fits')

eLIER = log[1].data['BPT_C'] == 3

plate = log[1].data['PLATE'][eLIER]
ifu = log[1].data['IFUDESIGN'][eLIER]

for i in range(len(plate)):
    maps_file = '/data/manga/spectro/analysis/MPL-7/HYB10-GAU-MILESHC/{0}/{1}/manga-{0}-{1}-MAPS-HYB10-GAU-MILESHC.fits.gz'.format(plate[i],ifu[i])
    if not os.path.isfile(maps_file):
        print('{0} does not exist'.format(maps_file))