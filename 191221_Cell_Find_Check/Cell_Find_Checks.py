# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 13:36:29 2020

@author: ZR

"""

import My_Wheels.Cross_Run_Align as Module_Align
import My_Wheels.List_Operation_Kit as List_Tools
import My_Wheels.Graph_Operation_Kit as Graph_Tools


#%% Step 1 Align
Day_Folder = [r'E:\Test_Data\200107_L80_LM']
Run_Folder = [
    '1-001',
    '1-002'
    ]

Run_In_Align = List_Tools.List_Annex(Day_Folder, Run_Folder)
CRA = Module_Align.Cross_Run_Align(Run_In_Align)
Align_Property = CRA.Do_Align()
#%% Step 2 Stim_Frame_Align
from My_Wheels.Stim_Frame_Align import Stim_Frame_Align
stim_folder = r'E:\Test_Data\200107_L80_LM\200107_L80_2P_stimuli\Run01_2P_G8'
_,Stim_Frame_Align = Stim_Frame_Align(stim_folder,head_extend = 0)

#%% Step3, On Off Map Generation.
Num_Run = 1
off_list = Stim_Frame_Align[-1]
on_list = []
on_stim_ids = list(Stim_Frame_Align.keys())
on_stim_ids.remove(-1)
for i in range(len(on_stim_ids)):
    on_list.extend(Stim_Frame_Align[on_stim_ids[i]])
sub_graph,dF_F = Graph_Tools.Graph_Subtractor(Align_Property['all_tif_name'][Num_Run],on_list,off_list,clip_std = 5)
Graph_Tools.Show_Graph(sub_graph,'On_Off_Graph',Align_Property['all_save_folders'][Num_Run])
print('dF/F Value is:'+str(dF_F))
#%% Step4, Cell find From Morphology.
from My_Wheels.Cell_Find_From_Graph import Cell_Find_From_Graph
import cv2
import My_Wheels.OS_Tools_Kit as OS_Tools
save_folder = r'E:\Test_Data\200107_L80_LM\1-001\Results'
Morphology_Cells = Cell_Find_From_Graph(cv2.imread(save_folder+r'\Global_Average_After_Align.tif',-1),find_thres = 2)
Morphology_Cells_Folder = save_folder + r'\Morphology_Cells'
OS_Tools.Save_Variable(Morphology_Cells_Folder,'Morphology_Cells',Morphology_Cells)
all_keys = list(Morphology_Cells.keys())
all_keys.remove('All_Cell_Information')
for i in range(len(all_keys)):
    Graph_Tools.Show_Graph(Morphology_Cells[all_keys[i]],graph_name = all_keys[i],save_path = save_folder+r'\Morphology_Cells',show_time = 0,write = True)

#%% Step 5, On Off 