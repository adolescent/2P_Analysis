# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 21:38:02 2021

@author: ZR
"""

from My_Wheels.Standard_Aligner import Standard_Aligner
import cv2
import OS_Tools_Kit as ot
import Graph_Operation_Kit as gt
import numpy as np
from Affine_Alignment import Affine_Core_Point_Equal

#%% Too tremble, manual align each run...
day_folder = r'G:\Test_Data\2P\211202_L86_2P'
Sa = Standard_Aligner(day_folder, [1,2,3,4,5],trans_range = 50)
Sa.One_Key_Aligner_No_Affine()
#%% Cross Run Align manually. Use Run04 as base.
all_result_folder = Sa.all_resultfolder
final_base = cv2.imread(all_result_folder[3]+r'\Run_Average_After_Align.tif',-1)
targ_mask = np.ones(shape = (512,512))
gt.Show_Graph((targ_mask*65535).astype('u2'), 'Location_Mask', all_result_folder[3])
folder_need_align = all_result_folder.tolist()
folder_need_align.pop(3)
translation_matix_dic = {}
for i,c_folder in enumerate(folder_need_align):
    c_avr_graph = cv2.imread(c_folder+r'\Run_Average_After_Align.tif',-1)
    c_matched_graph,c_h =  Affine_Core_Point_Equal(c_avr_graph, final_base,targ_gain = 1,good_match_prop= 0.1,sector_num=1)
    translation_matix_dic[c_folder.split('\\')[-2]] = c_h
    mask = np.ones(shape = (512,512))
    resized_mask = cv2.warpPerspective(mask, c_h, (512,512))
    gt.Show_Graph((resized_mask*65535).astype('u2'), 'Location_Mask', c_folder)
    resized_average = cv2.warpPerspective(c_avr_graph, c_h,(512,512))
    combined_graph = cv2.cvtColor(final_base,cv2.COLOR_GRAY2RGB).astype('f8')
    combined_graph[:,:,1] += resized_average
    combined_graph = np.clip(combined_graph,0,65535).astype('u2')
    gt.Show_Graph(combined_graph,'Combined_Location_Graph', c_folder)
    gt.Show_Graph(resized_average, 'Resized_Average', c_folder)
    atn = ot.Get_File_Name(c_folder+r'\Aligned_Frames')
    final_align_folder = c_folder+r'\Final_Aligned_Frames'
    for j,c_graph_path in enumerate(atn):
        c_graph_name = c_graph_path.split('\\')[-1][:-4]
        c_graph = cv2.imread(c_graph_path,-1)
        aligned_c_graph = cv2.warpPerspective(c_graph, c_h, (512,512))
        gt.Show_Graph(aligned_c_graph, c_graph_name, final_align_folder,show_time=0)
        
for i in range(len(all_result_folder)):
    after_atn = ot.Get_File_Name(all_result_folder[i]+r'\Final_Aligned_Frames')
    after_average = gt.Clip_And_Normalize(gt.Average_From_File(after_atn),clip_std=5)
    gt.Show_Graph(after_average, 'Final_Averaged_Graph', all_result_folder[i])
    
#%% Get Final average.
all_averaged_tif_name = []
for i in range(len(Sa.run_subfolder)):
    all_averaged_tif_name.extend(ot.Get_File_Name(Sa.all_resultfolder[i]+r'\Final_Aligned_Frames'))
total_averaged_graph = gt.Clip_And_Normalize(gt.Average_From_File(all_averaged_tif_name),clip_std=5)

for i in range(len(Sa.all_resultfolder)):
    if i ==0:
        gt.Show_Graph(total_averaged_graph, 'Global_Average', Sa.all_resultfolder[i])
    else:
        gt.Show_Graph(total_averaged_graph, 'Global_Average', Sa.all_resultfolder[i],show_time = 0)
    

#%% Get Basic Stim Maps
from Stim_Frame_Align import One_Key_Stim_Align
One_Key_Stim_Align(r'G:\Test_Data\2P\211202_L86_2P\211202_L86_stimuli')
from Standard_Stim_Processor import One_Key_Frame_Graphs
from Standard_Parameters.Sub_Graph_Dics import Sub_Dic_Generator
Hue_Sub_Dic = Sub_Dic_Generator('HueNOrien4',para = 'Default')
One_Key_Frame_Graphs(r'G:\Test_Data\2P\211202_L86_2P\1-004', Hue_Sub_Dic)
Shape_Para = {}
Shape_Para['Circle-Triangle'] = [list(range(25,32)),list(range(17,24))]
Shape_Para['Circle-Bar'] = [list(range(25,32)),list(range(1,8))]
Shape_Para['Triangle-Bar'] = [list(range(17,24)),list(range(1,8))]
Shape_Para['DirU-D_Circle'] = [[26,27],[30,31]]
Shape_Para['DirU-D_Triangle'] = [[18,19,20],[22,23,24]]
Shape_Para['DirU-D_Bar'] = [[2,3,4],[6,7,8]]
One_Key_Frame_Graphs(r'G:\Test_Data\2P\211202_L86_2P\1-006', Shape_Para)

G16_Para = {}
G16_Para['H-V'] = [[1,9],[5,13]]
G16_Para['A-O'] = [[3,11],[7,15]]
One_Key_Frame_Graphs(r'G:\Test_Data\2P\211202_L86_2P\1-007', G16_Para)
