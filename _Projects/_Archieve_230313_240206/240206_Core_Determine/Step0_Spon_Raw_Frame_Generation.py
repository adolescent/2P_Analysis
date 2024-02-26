
'''
This script will generate raw spon data from mmemap caiman data, and return spon rawdata matrix.

'''
#%%
import os
from Decorators import Timer
import time
import Graph_Operation_Kit as gt
import OS_Tools_Kit as ot
import List_Operation_Kit as lt
import Caiman_API.Pack_Graphs as Graph_Packer
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


wp= r'D:\_All_Spon_Data_V1\L76_18M_220902'
allgraph_name = r'D:\_All_Spon_Data_V1\L76_18M_220902\1-001_0000-#-7-#-6_d1_512_d2_512_d3_1_order_C_frames_29967.mmap'
c_spon_frame = ot.Load_Variable(wp,'Spon_Before.pkl')
spon_starttime = c_spon_frame.index[0]
spon_endtime = c_spon_frame.index[-1]

#%% Load all mme frame stacks.
# These operation are fast as they use memmap tech, but real calculation is slow, careful.
Yr, dims, T = cm.load_memmap(allgraph_name)
images_all = np.reshape(Yr.T, [T] + list(dims), order='F')
all_spon_before_frames = images_all[spon_starttime:spon_endtime+1,:,:]
#%% Save all spon frames. This is slow.
raw_before_spon_frames = np.array(all_spon_before_frames)
ot.Save_Variable(wp,'Spon_Before_Raw',raw_before_spon_frames)

