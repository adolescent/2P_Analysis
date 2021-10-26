# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 10:23:28 2021

@author: ZR
"""
#%% Align Run01 and all other runs.
from Standard_Aligner import Standard_Aligner
day_folder = r'F:\Test_Data\2P\211009_L76_2P'
Sa_1 = Standard_Aligner(day_folder, [1], trans_range=50)
Sa_1.One_Key_Aligner()
Sa_2 = Standard_Aligner(day_folder, [2, 3, 4, 5, 6, 7], trans_range=30, final_base='1-002')
Sa_2.One_Key_Aligner()

from Stim_Frame_Align import One_Key_Stim_Align
One_Key_Stim_Align(r'F:\Test_Data\2P\211009_L76_2P\211009_L76_stimuli')
from Standard_Parameters.Sub_Graph_Dics import Sub_Dic_Generator
from Standard_Stim_Processor import One_Key_Frame_Graphs
G16_Para = Sub_Dic_Generator('G16_2P')
One_Key_Frame_Graphs(r'F:\Test_Data\2P\211009_L76_2P\1-002', G16_Para)
Shape_Para = Sub_Dic_Generator('Shape3Dir8')
One_Key_Frame_Graphs(r'F:\Test_Data\2P\211009_L76_2P\1-006', Shape_Para)
RG_Para = Sub_Dic_Generator('RGLum4')
One_Key_Frame_Graphs(r'F:\Test_Data\2P\211009_L76_2P\1-007', RG_Para)

from Cell_Find_From_Graph import Cell_Find_From_Mannual
Manual_Cell = Cell_Find_From_Mannual(r'G:\Test_Data\2P\211009_L76_2P\_Manual_Cell\Cell_Mask.png',
                                     r'G:\Test_Data\2P\211009_L76_2P\_Manual_Cell\Global_Average.tif')

from Standard_Cell_Generator import Standard_Cell_Generator
Scg = Standard_Cell_Generator('L76', '211009', r'G:\Test_Data\2P\211009_L76_2P', [2,3,4,5,6,7])
Scg.Generate_Cells()

#%%Get Run01 Cells and compare to other runs.
Run01_Cells = Cell_Find_From_Mannual(r'G:\Test_Data\2P\211009_L76_2P\1-001\Results\Cells_Run01\Cell_Mask_For_Run01.png',
                                     average_graph_path=r'G:\Test_Data\2P\211009_L76_2P\1-001\Results\Cells_Run01\Run01_Average.tif')

import csv
csv_path = r'G:\Test_Data\2P\211009_L76_2P\1-001\Results\Cell_Compare.csv'
compare_list = []
with open(csv_path) as f:
    f_tsv = csv.reader(f, delimiter=',')
    headers = next(f_tsv)
    for row in f_tsv:
        compare_list.append(row)

import OS_Tools_Kit as ot
from Spike_Train_Generator import Spike_Train_Generator
Run01_Cell_Info = Run01_Cells['All_Cell_Information']
all_01_tif_name = ot.Get_File_Name(r'G:\Test_Data\2P\211009_L76_2P\1-001\Results\Final_Aligned_Frames')
all_01_F,all_01_dF = Spike_Train_Generator(all_01_tif_name, Run01_Cell_Info)
# Add new Run01 data to origin.
all_cells = ot.Load_Variable(r'G:\Test_Data\2P\211009_L76_2P','L76_211009A_All_Cells_Without_Run01.ac')
new_cell_dic = {}
for i in range(len(compare_list)):
    c_cellname = 'L76_211009A_'+ot.Bit_Filler(i)
    tc = all_cells[c_cellname]
    targeted_01_cellid = int(compare_list[i][1])
    if targeted_01_cellid == -1:
        tc['In_Run']['Run001'] = False
    else:
        tc['In_Run']['Run001'] = True
        tc['Run001'] = {}
        tc['Run001']['F_train'] = all_01_F[targeted_01_cellid]
        tc['Run001']['dF_train'] = all_01_dF[targeted_01_cellid]
        tc['Run001']['Mean_F'] = all_01_F[targeted_01_cellid].mean()
        tc['Run001']['STD_F'] = all_01_F[targeted_01_cellid].std()
    new_cell_dic[c_cellname] = tc
ot.Save_Variable(r'G:\Test_Data\2P\211009_L76_2P', 'L76_210721A_All_Cell_Include_Run01', new_cell_dic,'.ac')

