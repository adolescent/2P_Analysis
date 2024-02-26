'''
This script will cut out example location's response, this will be useful on showing raw datas.

'''

#%%
import os
from Decorators import Timer
import time
import Graph_Operation_Kit as gt
import OS_Tools_Kit as ot
import List_Operation_Kit as lt
import Caiman_API.Pack_Graphs as Graph_Packer
from Cell_Class.Stim_Calculators import Stim_Cells
from Cell_Class.Format_Cell import Cell
import bokeh.plotting as bpl
from tqdm import tqdm
import cv2
import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params
from caiman.utils.utils import download_demo
from caiman.utils.visualization import plot_contours, nb_view_patches, nb_plot_contour
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import skimage.io
from caiman.source_extraction.cnmf.cnmf import load_CNMF

wp = r'D:\_All_Spon_Data_V1\L76_18M_220902'
allgraph_name = r'D:\_All_Spon_Data_V1\L76_18M_220902\1-001_0000-#-7-#-6_d1_512_d2_512_d3_1_order_C_frames_29967.mmap'
c_spon_frame = ot.Load_Variable(wp,'Spon_Before.pkl')
spon_starttime = c_spon_frame.index[0]
spon_endtime = c_spon_frame.index[-1]
ac = ot.Load_Variable_v2(wp,'Cell_Class.pkl')

#%%
Yr, dims, T = cm.load_memmap(allgraph_name)
images_all = np.reshape(Yr.T, [T] + list(dims), order='F')

r1 = len(ac.Z_Frames['1-001'])
r2 = len(ac.Z_Frames['1-002'])
r3 = len(ac.Z_Frames['1-003'])
r4 = len(ac.Z_Frames['1-006'])
r5 = len(ac.Z_Frames['1-007'])
r6 = len(ac.Z_Frames['1-008'])

all_spon_before_frames = images_all[spon_starttime:spon_endtime+1,:,:]
all_od_frames = images_all[(r1+r2+r3):(r1+r2+r3+r4+1),:,:]
all_orien_frames = images_all[(r1+r2+r3+r4):(r1+r2+r3+r4+r5+1),:,:]
all_color_frames = images_all[(r1+r2+r3+r4+r5):(r1+r2+r3+r4+r5+r6+1),:,:]

#%% Save all spon frames. This is slow.
# raw_before_spon_frames = np.array(all_spon_before_frames)
# ot.Save_Variable(wp,'Spon_Before_Raw',raw_before_spon_frames)
ot.Save_Variable(wp,'OD_Frames_Raw',np.array(all_od_frames))
ot.Save_Variable(wp,'Orien_Frames_Raw',np.array(all_orien_frames))
ot.Save_Variable(wp,'Color_Frames_Raw',np.array(all_color_frames))


