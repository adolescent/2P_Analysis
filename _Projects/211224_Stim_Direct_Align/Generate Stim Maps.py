# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 10:25:16 2021

@author: ZR
"""

import OS_Tools_Kit as ot
import cv2
from Affine_Alignment import Affine_Core_Point_Equal
import numpy as np
import Graph_Operation_Kit as gt


#%% Align 0604 cells to 0628 frame.
base_graph = cv2.imread(r'G:\Test_Data\2P\210629_L76_2P\Global_Average.tif',-1)
target_graph = cv2.imread(r'G:\Test_Data\2P\210629_L76_2P\1-099\Results\Before_Averaged_Graph.tif',-1)
c_matched_graph,c_h =  Affine_Core_Point_Equal(target_graph,base_graph,targ_gain = 1)
all_tif_name = ot.Get_File_Name(r'G:\Test_Data\2P\210629_L76_2P\1-099\Results\Before_Aligned_Frames')
save_folder = r'G:\Test_Data\2P\210629_L76_2P\1-099\Results\Final_Aligned_Frames'


mask = np.ones(shape = (512,512))
resized_mask = cv2.warpPerspective(mask, c_h, (512,512))
gt.Show_Graph((resized_mask*65535).astype('u2'), 'Location_Mask', save_folder)
# Combined graph, for record
resized_average = cv2.warpPerspective(target_graph, c_h,(512,512))
combined_graph = cv2.cvtColor(base_graph,cv2.COLOR_GRAY2RGB).astype('f8')
combined_graph[:,:,1] += resized_average
combined_graph = np.clip(combined_graph,0,65535).astype('u2')
gt.Show_Graph(combined_graph, 'Combined_Location_Graph', save_folder)
# Then get final affined graph, based on former 

for i,cname in enumerate(all_tif_name):
    c_graph_name = cname.split('\\')[-1][:-4]
    c_graph = cv2.imread(cname,-1)
    aligned_c_graph = cv2.warpPerspective(c_graph, c_h, (512,512))
    gt.Show_Graph(aligned_c_graph, c_graph_name, save_folder,show_time=0)

final_tif_name = ot.Get_File_Name(r'G:\Test_Data\2P\210629_L76_2P\1-099\Results\Final_Aligned_Frames')
final_avr = gt.Average_From_File(final_tif_name)
final_avr = gt.Clip_And_Normalize(final_avr,clip_std = 5)
cv2.imwrite(save_folder+r'\Final_Averaged_Graph.tif',final_avr)


#%% Regenerate cells
from Standard_Cell_Generator import Standard_Cell_Generator
Scg = Standard_Cell_Generator('L76', '210629', r'G:\Test_Data\2P\210629_L76_2P',[1,2,3,4,5,6,7,99])
Scg.Generate_Cells()
