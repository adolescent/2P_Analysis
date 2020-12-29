# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 11:35:32 2020

@author: ZR
"""

import My_Wheels.ROI_Matcher as Matcher
import cv2
import My_Wheels.Graph_Operation_Kit as Graph_Tools
import numpy as np
import My_Wheels.OS_Tools_Kit as OS_Tools
import My_Wheels.Graph_Selector as Selector
#%% Get ROI 
ROI_graph = cv2.imread(r'I:\Test_Data\2P\201222_L76_2P\ROI_Analyzer\ROI_Run01.tif',-1)
ful_graph_today = cv2.imread(r'I:\Test_Data\2P\201222_L76_2P\ROI_Analyzer\1222_Full_Graph_Run05.tif',-1)
ful_graph_prev = cv2.imread(r'I:\Test_Data\2P\201222_L76_2P\ROI_Analyzer\1211_Full_Graph_Run01.tif',-1)
#%% Match today's data
save_folder = r'I:\Test_Data\2P\201222_L76_2P\ROI_Analyzer'
merged_graph_today,loc_graph_today,ROI_boulder_today = Matcher.ROI_Matcher(ful_graph_today, ROI_graph)
Graph_Tools.Show_Graph(merged_graph_today, 'Today_Merged', save_folder)
Graph_Tools.Show_Graph(loc_graph_today, 'Today_Local', save_folder)
#%% Then Match last day.
merged_graph_prev,loc_graph_prev,ROI_boulder_prev = Matcher.ROI_Matcher(ful_graph_prev, ROI_graph)
Graph_Tools.Show_Graph(merged_graph_prev, 'Prev_Merged', save_folder)
Graph_Tools.Show_Graph(loc_graph_prev, 'Prev_Local', save_folder)
#%% Then match ROI data with stim full graph data.
OD_Graph = cv2.imread(r'I:\Test_Data\2P\201211_L76_2P\1-012\Results\Subtraction_Graphs\OD_SubGraph.tif').astype('f8')
OD_Graph[:,:,2] += ROI_boulder_prev
OD_Graph_Annotate = np.clip(OD_Graph,0,255).astype('u1')
Graph_Tools.Show_Graph(OD_Graph_Annotate, 'OD_Annotate', save_folder)
#%% Then H-V
HV_Graph = cv2.imread(r'I:\Test_Data\2P\201211_L76_2P\1-010\Results\Subtraction_Graphs\H-V_SubGraph.tif').astype('f8')
HV_Graph[:,:,2] += ROI_boulder_prev
HV_Graph_Annotate = np.clip(HV_Graph,0,255).astype('u1')
Graph_Tools.Show_Graph(HV_Graph_Annotate, 'HV_Annotate', save_folder)
#%% Then A-O
AO_Graph = cv2.imread(r'I:\Test_Data\2P\201211_L76_2P\1-010\Results\Subtraction_Graphs\A-O_SubGraph.tif').astype('f8')
AO_Graph[:,:,2] += ROI_boulder_prev
AO_Graph_Annotate = np.clip(AO_Graph,0,255).astype('u1')
Graph_Tools.Show_Graph(AO_Graph_Annotate, 'AO_Annotate', save_folder)
#%% Then ALl-0
All0_Graph = cv2.imread(r'I:\Test_Data\2P\201211_L76_2P\1-010\Results\Subtraction_Graphs\All-0_SubGraph.tif').astype('f8')
All0_Graph[:,:,2] += ROI_boulder_prev
All0_Graph_Annotate = np.clip(All0_Graph,0,255).astype('u1')
Graph_Tools.Show_Graph(All0_Graph_Annotate, 'All-0_Annotate', save_folder)
#%% Last, RG-Lum
RG_Graph = cv2.imread(r'I:\Test_Data\2P\201211_L76_2P\1-014\Results\Subtraction_Graphs\RG-Lum_SubGraph.tif').astype('f8')
RG_Graph[:,:,2] += ROI_boulder_prev
RG_Graph_Annotate = np.clip(RG_Graph,0,255).astype('u1')
Graph_Tools.Show_Graph(RG_Graph_Annotate, 'RG_Annotate', save_folder)
#%% Next job, video compare.
import My_Wheels.Video_Writer as Video_Writer
roi_spon_folder = r'I:\Test_Data\2P\201222_L76_2P\1-001\Results\Aligned_Frames'
Video_Writer.Video_From_File(roi_spon_folder,(325,324),fps = 30,cut_boulder = [0,0,0,0])
full_frame_folder = r'I:\Test_Data\2P\201211_L76_2P\1-001\Results\Aligned_Frames'
Video_Writer.Video_From_File(full_frame_folder,(492,492),fps = 15,cut_boulder = [10,10,10,10])
#%% Then find a exp cell of ROI and full graph.
roi_cell_dic = OS_Tools.Load_Variable(r'I:\Test_Data\2P\201222_L76_2P\1-008\Results\Global_Morpho','Global_Morpho.cell')
ROI_G8_F_Train = OS_Tools.Load_Variable(r'I:\Test_Data\2P\201222_L76_2P\1-010\Results\F_Trains.pkl')
ROI_G8_Stim_Dic = OS_Tools.Load_Variable(r'I:\Test_Data\2P\201222_L76_2P\1-010\Results\Stim_Frame_Align.pkl')

full_cell_dic = OS_Tools.Load_Variable(r'I:\Test_Data\2P\201211_L76_2P\1-010\Results\Global_Morpho\Global_Morpho.cell')
full_G8_F_Train = OS_Tools.Load_Variable(r'I:\Test_Data\2P\201211_L76_2P\1-010\Results\F_train.pkl')
full_Stim_Dic = OS_Tools.Load_Variable(r'I:\Test_Data\2P\201211_L76_2P\1-010\Results\Stim_Frame_Align.pkl')
#%% Run01 Bright and Low
aligned_folder = r'I:\Test_Data\2P\201222_L76_2P\1-001\Results\Aligned_Frames'
save_path = r'I:\Test_Data\2P\201222_L76_2P\1-001\Results'
bright_graph,_ = Selector.Intensity_Selector(aligned_folder,list_write= False)
bright_graph = np.clip(bright_graph.astype('f8')*32,0,65535).astype('u2')
Graph_Tools.Show_Graph(bright_graph, 'Brightest_Graph', save_path)

low_graph,_ = Selector.Intensity_Selector(aligned_folder,list_write= False,mode = 'smallest')
low_graph = np.clip(low_graph.astype('f8')*32,0,65535).astype('u2')
Graph_Tools.Show_Graph(low_graph, 'Darkest_Graph', save_path)