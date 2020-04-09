# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 13:36:29 2020

@author: ZR

"""

import My_Wheels.Cross_Run_Align as Module_Align
import My_Wheels.List_Operation_Kit as List_Tools
import My_Wheels.Graph_Operation_Kit as Graph_Tools
import My_Wheels.OS_Tools_Kit as OS_Tools

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
stim_folder = r'E:\Test_Data\200107_L80_LM\200107_L80_2P_stimuli\Run02_2P_manual_OD8'
_,Stim_Frame_Align = Stim_Frame_Align(stim_folder,head_extend = -2)

#%% Step3, On Off Map Generation.
Num_Run = 1
off_list = Stim_Frame_Align[-1]
on_list = []
on_stim_ids = list(Stim_Frame_Align.keys())
on_stim_ids.remove(-1)
for i in range(len(on_stim_ids)):
    on_list.extend(Stim_Frame_Align[on_stim_ids[i]])
sub_graph,dF_F = Graph_Tools.Graph_Subtractor(all_tif_name,on_list,off_list,clip_std = 5)
#Graph_Tools.Show_Graph(sub_graph,'On_Off_Graph',Align_Property['all_save_folders'][Num_Run])
print('dF/F Value is:'+str(dF_F))
#%% Step4, Cell find From Morphology and On-Off
from My_Wheels.Cell_Find_From_Graph import Cell_Find_And_Plot
Morphology_Graph_Name = r'Global_Average_After_Align.tif'
Morphology_Cells = Cell_Find_And_Plot(save_folder,Morphology_Graph_Name,'Morphology_Cells',find_thres = 2)
On_Off_Graph_Name = r'On_Off_Graph.tif'
On_Off_Cells = Cell_Find_And_Plot(save_folder,On_Off_Graph_Name,'On_Off_Cells',find_thres = 2)

#%% Step 5, Get Compare Matrix.
import My_Wheels.Cell_Visualization as Vi
Compare_Matrix = Vi.Cell_Information_Compare(Run1_Onoff,Run2_Onoff,shift_limit = 15,plot = True, save_folder = save_folder)

    