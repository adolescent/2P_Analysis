# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 13:41:07 2020

@author: ZR
Codes to process L76 Data
"""

import My_Wheels.Graph_Operation_Kit as Graph_Tools
import My_Wheels.OS_Tools_Kit as OS_Tools
#%% Cell1, Average Graph.
graph_folder = r'I:\Test_Data\201023_L76_LM\1-003'
save_path = graph_folder+r'\Results'
OS_Tools.mkdir(save_path)
all_tif_name = OS_Tools.Get_File_Name(graph_folder)
average_graph = Graph_Tools.Average_From_File(all_tif_name)
norm_average_graph = Graph_Tools.Clip_And_Normalize(average_graph,clip_std = 3)
Graph_Tools.Show_Graph(norm_average_graph, 'Average_Graph',save_path)
#%% Then Calculate Runs
graph_folder = r'I:\Test_Data\201023_L76_LM\1-013'
import My_Wheels.Translation_Align_Function as Align
Align.Translation_Alignment([graph_folder])
#%% Align Stim and Frame
import My_Wheels.Stim_Frame_Align as Stim_Frame_Align
stim_folder = r'I:\Test_Data\201023_L76_LM\201023_L76_LM_Stimulus\Run13_RGLum4'
Frame_Stim_Sequence,Frame_Stim_Dictionary = Stim_Frame_Align.Stim_Frame_Align(stim_folder)
aligned_tif_name = OS_Tools.Get_File_Name(r'I:\Test_Data\201023_L76_LM\1-013\Results\Aligned_Frames')
#%% Generate On-Off Map
on_id = []
on_id.extend(Frame_Stim_Dictionary[3])
on_id.extend(Frame_Stim_Dictionary[4])
off_id =[]
off_id.extend(Frame_Stim_Dictionary[0])
#off_id.extend(Frame_Stim_Dictionary[4])

on_graph_name = []
for i in range(len(on_id)):
    on_graph_name.append(aligned_tif_name[on_id[i]])
off_graph_name = []
for i in range(len(off_id)):
    off_graph_name.append(aligned_tif_name[off_id[i]])
    
on_graph = Graph_Tools.Average_From_File(on_graph_name).astype('f8')
off_graph = Graph_Tools.Average_From_File(off_graph_name).astype('f8')
on_off_graph = on_graph-off_graph
norm_on_off_graph = Graph_Tools.Clip_And_Normalize(on_off_graph,clip_std = 2.5)
Graph_Tools.Show_Graph(norm_on_off_graph,'Lum-0_Graph',r'I:\Test_Data\201023_L76_LM\1-013\Results')