# -*- coding: utf-8 -*-
"""
Created on Tue May 17 17:04:24 2022

@author: adolescent

This script is used to generate numbered caiman mask for given data. 
Further version of caiman will do this work autoamtically.

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
import OS_Tools_Kit as ot

import caiman as cm
from caiman.source_extraction.cnmf.cnmf import load_CNMF
from caiman.source_extraction.cnmf.params import CNMFParams
#%% Load and get used cells.
caiman_folder = r'D:\Test_Data\2P\220421_L85\_CAIMAN'
cnm2 = load_CNMF(caiman_folder+r'\analysis_results.hdf5')
# generate counter maps.
raw_cells = cnm2.estimates.idx_components
cnm2.estimates.plot_contours_nb(idx = raw_cells)
est = cnm2.estimates
used_cells = []
x_range = (20,492)
y_range = (20,492)
for i,c_comp in enumerate(raw_cells):
    c_loc = cnm2.estimates.coordinates[c_comp]['CoM']
    if (c_loc[0]>x_range[0] and c_loc[0]<x_range[1]) and (c_loc[1]>y_range[0] and c_loc[1]<y_range[1]):
        used_cells.append(c_comp)
comp_id_dic = {}
for i in range(380):
    comp_id_dic[i+1] = used_cells[i]
ot.Save_Variable(caiman_folder, 'comp_id_dic', comp_id_dic)
#%% Plot graphs.
import numpy as np
import My_Wheels.Graph_Operation_Kit as gt
import cv2
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

est = cnm2.estimates
all_cell_ids = used_cells
base_graph = cv2.imread(r'D:\Test_Data\2P\220421_L85\Global_Average.tif')
font = ImageFont.truetype('arial.ttf',11)
annotated_graph = np.zeros(shape = (512,512,3),dtype = 'f8')
for i,c_id in enumerate(all_cell_ids):
    annotated_graph[:,:,0] += np.reshape(cnm2.estimates.A[:,c_id].toarray(), (512,512), order='F')*100
    annotated_graph[:,:,1] += np.reshape(cnm2.estimates.A[:,c_id].toarray(), (512,512), order='F')*100
    annotated_graph[:,:,2] += np.reshape(cnm2.estimates.A[:,c_id].toarray(), (512,512), order='F')*100
annotated_graph = gt.Clip_And_Normalize(annotated_graph,clip_std = 8,bit = 'u1')
im = Image.fromarray(annotated_graph)
for i,c_id in enumerate(all_cell_ids):
    y,x = cnm2.estimates.coordinates[c_id]['CoM']
    draw = ImageDraw.Draw(im)
    draw.text((x+5,y+5),str(i+1),(0,255,100),font = font,align = 'center')
final_graph = np.array(im)
gt.Show_Graph(final_graph, 'Numbers', caiman_folder) 


