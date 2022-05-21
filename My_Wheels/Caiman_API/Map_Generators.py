# -*- coding: utf-8 -*-
"""
Created on Mon May 16 13:04:25 2022

@author: adolescent
"""



import os
from Decorators import Timer
import time
import Graph_Operation_Kit as gt
import OS_Tools_Kit as ot
import os
import List_Operation_Kit as lt
from Caiman_API.Pack_Graphs import Graph_Packer
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
from caiman.source_extraction.cnmf.params import CNMFParams
#%%
class Map_Generator(object):
    
    name = 'Generate Subtraction maps'
    
    def __init__(self,day_folder,sub_folder = '_CAIMAN',series_dic_name = 'All_Series_Dic.pkl'):
        
        self.workpath = ot.join(day_folder,sub_folder)
        self.cnm2_file = load_CNMF(ot.Get_File_Name(self.workpath,'.hdf5')[0])
        self.cell_dics = ot.Load_Variable(ot.join(self.workpath,series_dic_name))
        self.Stim_Frame_Align = ot.Load_Variable(ot.join(day_folder,'_All_Stim_Frame_Infos.sfa'))
        
    def Single_T_Map(self,run_folder,p_thres = 0.05,used_frame = [4,5]):
        pass
        
        







#%%
if __name__ == '__main__':
        
    day_folder = r'D:\Test_Data\2P\220421_L85'
# =============================================================================
#     #change mask file in celldics.
#     compare_dic = ot.Load_Variable(r'D:\Test_Data\2P\220421_L85\_CAIMAN\comp_id_dic.pkl')
#     acn = list(compare_dic.keys())
#     est = cnm2_file.estimates
#     for i,cc in enumerate(acn):
#         c_mask_new = np.reshape(est.A[:,compare_dic[cc]].toarray(), (512,512), order='F')
#         cell_dics[cc]['Cell_Mask'] = c_mask_new
#     ot.Save_Variable(r'D:\Test_Data\2P\220421_L85\_CAIMAN', 'All_Series_Dic.pkl', cell_dics)
# =============================================================================

