# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 21:33:19 2020

@author: adolescent
All codes below is very speficied. Only promised on 201111-L76,This file is for parameter archieve.
"""
from My_Wheels.Translation_Align_Function import Translation_Alignment
import My_Wheels.List_Operation_Kit as List_Tools
import My_Wheels.OS_Tools_Kit as OS_Tools
#%% Cell1 Align part. We use first 3 run and latter ones to match .Base graph is Run1-002 Before Average.
data_folder = r'E:\Test_Data\2P\201111_L76_LM'
run_folder =['1-001','1-002','1-003','1-009','1-012','1-013']
all_folders = List_Tools.List_Annex([data_folder], run_folder)
Translation_Alignment(all_folders,base_mode=1,align_range=35,align_boulder=35)
'''Attention here,1-012 and 1-013 have more movement than 20pix, making this hard to use.'''

#%% Align Stim and frame of each stim folder.
from My_Wheels.Stim_Frame_Align import Stim_Frame_Align
all_stim_folders = [
    r'G:\Test_Data\2P\201111_L76_LM\201111_L76_2P_stimuli\Run02_2P_G8',
    r'G:\Test_Data\2P\201111_L76_LM\201111_L76_2P_stimuli\Run03_2P_manual_OD8',
    r'G:\Test_Data\2P\201111_L76_LM\201111_L76_2P_stimuli\Run07_2P_RGLum4',
    r'G:\Test_Data\2P\201111_L76_LM\201111_L76_2P_stimuli\Run08_2P_RGLum4_RG',
    r'G:\Test_Data\2P\201111_L76_LM\201111_L76_2P_stimuli\Run09_2P_RGLum4',
    ]
for i in range(len(all_stim_folders)):
    current_stim_folder = all_stim_folders[i]
    _,current_Stim_Frame_Align = Stim_Frame_Align(current_stim_folder)
    OS_Tools.Save_Variable(current_stim_folder, 'Stim_Frame_Align', current_Stim_Frame_Align)
#%% Cell Find.
from My_Wheels.Cell_Find_From_Graph import On_Off_Cell_Finder
import My_Wheels.Graph_Operation_Kit as Graph_tools
def Cell_Find(run_folder):
    output_folder = run_folder+r'\Results'
    aligned_frame_folder = output_folder+r'\Aligned_Frames'
    all_tif_name = OS_Tools.Get_File_Name(aligned_frame_folder)
    Stim_Frame_Dic = OS_Tools.Load_Variable(output_folder,'Stim_Frame_Align.pkl')
    on_off_graph,Finded_Cells = On_Off_Cell_Finder(all_tif_name, Stim_Frame_Dic,shape_boulder=[20,20,20,35],filter_method = 'Gaussian',LP_Para = ((5,5),1.5))
    cell_folder = output_folder+r'\Cells'
    OS_Tools.Save_Variable(cell_folder, 'Finded_Cells', Finded_Cells,'.cell')
    Graph_tools.Show_Graph(on_off_graph, 'on-off_graph', cell_folder)
    all_keys = list(Finded_Cells.keys())
    all_keys.remove('All_Cell_Information')
    for i in range(len(all_keys)):
        Graph_tools.Show_Graph(Finded_Cells[all_keys[i]], all_keys[i], cell_folder)
    return True
run_list = [
    r'G:\Test_Data\2P\201111_L76_LM\1-002',
    r'G:\Test_Data\2P\201111_L76_LM\1-003',
    r'G:\Test_Data\2P\201111_L76_LM\1-009'
    ]
for i in range(3):
    Cell_Find(run_list[i])
#%% Calculate spike train of all finded cells.
from My_Wheels.Spike_Train_Generator import Spike_Train_Generator
run_list = [
    r'G:\Test_Data\2P\201111_L76_LM\1-002',
    r'G:\Test_Data\2P\201111_L76_LM\1-003',
    r'G:\Test_Data\2P\201111_L76_LM\1-009'
    ]
for i in range(3):
    cell_dic = OS_Tools.Load_Variable(run_list[i]+r'\Results\Cells\Finded_Cells.cell')
    all_tif_name = OS_Tools.Get_File_Name(run_list[i]+r'\Results\Aligned_Frames')
    stim_train = OS_Tools.Load_Variable(run_list[i]+r'\Results\Stim_Frame_Align.pkl')['Original_Stim_Train']
    F_train,dF_F_train = Spike_Train_Generator(all_tif_name, cell_dic['All_Cell_Information'])
    OS_Tools.Save_Variable(run_list[i]+r'\Results\Cells', 'F_train', F_train)
    OS_Tools.Save_Variable(run_list[i]+r'\Results\Cells', 'dF_F_train', dF_F_train)
#%% Calculate subgraph one by one.
from My_Wheels.Standard_Parameters.Sub_Graph_Dics import Sub_Dic_Generator
from My_Wheels.Standard_Stim_Processor import Standard_Stim_Processor

G8_Subdic = Sub_Dic_Generator('G8+90')
Standard_Stim_Processor(r'G:\Test_Data\2P\201111_L76_LM\1-002',
                        stim_folder = r'G:\Test_Data\2P\201111_L76_LM\1-002\Results\Stim_Frame_Align.pkl',
                        sub_dic = G8_Subdic,
                        tuning_graph=False,
                        cell_method = r'G:\Test_Data\2P\201111_L76_LM\1-002\Results\Cells\Finded_Cells.cell',
                        spike_train_path=r'G:\Test_Data\2P\201111_L76_LM\1-002\Results\Cells\dF_F_train.pkl',
                        )
#%% Then OD

#%% Then RGLum4