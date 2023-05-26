# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 15:20:15 2022

Use CAIMAN, so make sure use the right environment.
@author: ZR
"""

from Decorators import Timer
import time
import Graph_Operation_Kit as gt
import List_Operation_Kit as lt
import OS_Tools_Kit as ot
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

#%% Make Run04/Run05 as tif stacks.
from Caiman_API.Pack_Graphs import Graph_Packer
day_folder = [r'G:\Test_Data\2P\220415_L76']
run_sub_folder = ['1-004','1-005']
folder_lists = lt.List_Annex(day_folder, run_sub_folder)
work_folder = day_folder[0]+r'\_CAIMAN'
ot.mkdir(work_folder)
graph_nums = Graph_Packer(folder_lists, work_folder)
repacked_filename = ot.Get_File_Name(work_folder)

