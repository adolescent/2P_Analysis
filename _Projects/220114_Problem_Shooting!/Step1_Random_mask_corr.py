# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 14:32:34 2022

@author: adolescent
"""

import OS_Tools_Kit as ot
import cv2
from Series_Analyzer.Spontaneous_Preprocessing import Pre_Processor
from Series_Analyzer.Cell_Frame_PCA import Do_PCA,PCA_Regression
import matplotlib.pyplot as plt
from Series_Analyzer.Single_Component_Visualize import Single_Mask_Visualize
from Stimulus_Cell_Processor.Get_Tuning import Get_Tuned_Cells
import scipy.stats as stats
import numpy as np
#%% Read in mask and all cells.
day_folder = r'G:\Test_Data\2P\210831_L76_2P'
Run01_Frame = Pre_Processor(day_folder,start_time = 7000)
mask_folder = ot.Get_File_Name(r'F:\OneDrive\masks','.bmp')
all_cell_dic = ot.Load_Variable(day_folder,r'L76_210831A_All_Cells.ac')
masks = {}
for i,c_path in enumerate(mask_folder):
    masks[i] = cv2.imread(c_path,-1)
comp,info,weight = Do_PCA(Run01_Frame)
Run01_Frame_regressed = PCA_Regression(comp, info, weight,var_ratio = 0.75)
del comp,info,weight,mask_folder,i,c_path
acn = list(Run01_Frame.index)
tuning_dic = ot.Load_Variable(day_folder,r'All_Tuning_Property.tuning')
LE_cells = list(set(Get_Tuned_Cells(day_folder, 'LE',0.01))&set(acn))
RE_cells = list(set(Get_Tuned_Cells(day_folder, 'RE',0.01))&set(acn))
LE_cells.sort()
RE_cells.sort()

#%% Get LE/RE plots
LE_avr = Run01_Frame_regressed.loc[LE_cells,:].mean()
RE_avr = Run01_Frame_regressed.loc[RE_cells,:].mean()
# and masked plots too.
masked_plots = {}
for i in range(6):
    c_mask = masks[i]
    cells_in_mask = []
    cells_out_mask = []
    for j,cc in enumerate(acn):
        cc_cen = all_cell_dic[cc]['Cell_Info'].centroid
        if c_mask[int(cc_cen[0]),int(cc_cen[1])] == 0:
            cells_out_mask.append(cc)
        else:
            cells_in_mask.append(cc)
    in_mask_avr = Run01_Frame_regressed.loc[cells_in_mask,:].mean()
    out_mask_avr = Run01_Frame_regressed.loc[cells_out_mask,:].mean()
    masked_plots[i] = (in_mask_avr,out_mask_avr)
del i,j,cc,cc_cen,c_mask,cells_in_mask,cells_out_mask,in_mask_avr,out_mask_avr
corrs = np.zeros(6)
for i in range(6):
    corrs[i],_ = stats.pearsonr(masked_plots[i][0],masked_plots[i][1])



    


