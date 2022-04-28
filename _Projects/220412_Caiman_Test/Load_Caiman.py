# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 16:35:00 2022

@author: ZR
"""


import cv2
from datetime import datetime
from dateutil.tz import tzlocal
import numpy as np
import pyqtgraph as pg
import scipy
import os
from pyqtgraph import FileDialog
from pyqtgraph.Qt import QtGui
from pyqtgraph.parametertree import Parameter, ParameterTree
from scipy.sparse import csc_matrix

import caiman as cm
from caiman.source_extraction.cnmf.cnmf import load_CNMF
from caiman.source_extraction.cnmf.params import CNMFParams

#%%

day_folder = r'G:\Test_Data\2P\222222_L76_Fake_Data_For_Caiman'
cnm2 = load_CNMF(day_folder+r'\analysis_results.hdf5')
cnm2.mmap_file = day_folder+r'\memmap__d1_512_d2_512_d3_1_order_C_frames_945_.mmap'

Yr, dims, T = cm.load_memmap(cnm2.mmap_file)
images = np.reshape(Yr.T, [T] + list(dims), order='F') 
Cn = cm.local_correlations(images.transpose(1,2,0))
Cn[np.isnan(Cn)] = 0
cnm2.estimates.plot_contours_nb(img=Cn)


#%% Extract DF/F values
cnm2.estimates.detrend_df_f(quantileMin=8, frames_window=250)
#cnm2.estimates.nb_view_components(img=Cn, denoised_color='red')# This will include not passed comp in.
cnm2.estimates.nb_view_components(img=Cn, denoised_color='red',idx = cnm2.estimates.idx_components)
import matplotlib.pyplot as plt
plt.figure()
#To view the spatial components, their corresponding vectors need first to be reshaped into 2d images. For example if you want to view the i-th component you can type
# This will get component i's spatial map.
plt.imshow(np.reshape(cnm2.estimates.A[:,0].toarray(), dims, order='F'))
# this will get components i's temperal map.
temperal_series = cnm2.estimates.C[0,:]
dF_F = cnm2.estimates.F_dff[0,:]*1000
plt.plot(temperal_series)
plt.plot(dF_F)
