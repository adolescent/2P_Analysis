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
cnm_obj = load_CNMF(day_folder+r'\analysis_results.hdf5')
cnm_obj.mmap_file = day_folder+r'\memmap__d1_512_d2_512_d3_1_order_C_frames_945_.mmap'

Yr, dims, T = cm.load_memmap(cnm_obj.mmap_file)

denoised = cm.movie(cnm_obj.estimates.A.dot(cnm_obj.estimates.C) + \
                    cnm_obj.estimates.b.dot(cnm_obj.estimates.f)).reshape(dims + (-1,), order='F').transpose([2, 0, 1])
    
cnm_obj.estimates.W = None
cnm_obj.estimates.play_movie(denoised,save_movie = True)
