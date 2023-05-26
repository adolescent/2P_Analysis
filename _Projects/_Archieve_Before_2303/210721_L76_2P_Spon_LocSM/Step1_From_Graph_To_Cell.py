# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 15:22:26 2021

@author: ZR
"""

from Standard_Aligner import Standard_Aligner

Sa = Standard_Aligner(r'K:\Test_Data\2P\210721_L76_2P', [1,2,4,5,6,7])
Sa.One_Key_Aligner()
#%% Run 03 is not correct, Align seperately.
from Translation_Align_Function import Translation_Alignment
Translation_Alignment([r'K:\Test_Data\2P\210721_L76_2P\1-003'])
#%% Get all stim data.
from Stim_Frame_Align import One_Key_Stim_Align
One_Key_Stim_Align(r'K:\Test_Data\2P\210721_L76_2P\210721_stimuli')
#%% Get Stim Maps
from Standard_Stim_Processor import One_Key_Frame_Graphs
from Standard_Parameters.Sub_Graph_Dics import Sub_Dic_Generator
G16_Para = Sub_Dic_Generator('G16_2P')
One_Key_Frame_Graphs(r'K:\Test_Data\2P\210721_L76_2P\1-002',G16_Para)
OD_Para = Sub_Dic_Generator('OD_2P')
One_Key_Frame_Graphs(r'K:\Test_Data\2P\210721_L76_2P\1-006',OD_Para)
Hue_Para = Sub_Dic_Generator('HueNOrien4',para = {'Hue':['Red','Yellow','Green','Cyan','Blue','Purple','White']})
One_Key_Frame_Graphs(r'K:\Test_Data\2P\210721_L76_2P\1-007',Hue_Para)
#%% Get Cell From Manual
from Cell_Find_From_Graph import Cell_Find_From_Mannual
All_Cell = Cell_Find_From_Mannual(r'K:\Test_Data\2P\210721_L76_2P\_Manual_Cell\Cell_Mask.png',
                                  average_graph_path=r'K:\Test_Data\2P\210721_L76_2P\_Manual_Cell\Global_Average.tif',boulder = 15)
#%% Get All Cell Dic (except Run03)
from Standard_Cell_Generator import Standard_Cell_Generator
Scg = Standard_Cell_Generator('L76', '210721', r'K:\Test_Data\2P\210721_L76_2P', [1,2,4,5,6,7])
Scg.Generate_Cells()
#%% Get Run03 Cells & Trains
Run03_Cells = Cell_Find_From_Mannual(r'K:\Test_Data\2P\210721_L76_2P\1-003\Results\Cells_Run03\Cell_Mask_For_Run03.png',
                                     average_graph_path=r'K:\Test_Data\2P\210721_L76_2P\1-003\Results\Cells_Run03\Run03_Average.tif')
# Calculate F & dF Trains in specific run.
from Spike_Train_Generator import Spike_Train_Generator
import OS_Tools_Kit as ot
Run03_Cell_Info = Run03_Cells['All_Cell_Information']
all_03_tif_name = ot.Get_File_Name(r'K:\Test_Data\2P\210721_L76_2P\1-003\Results\Aligned_Frames')
all_03_F,all_03_dF = Spike_Train_Generator(all_03_tif_name, Run03_Cell_Info)
# Read in cell compare data.
import csv
csv_path = r'K:\Test_Data\2P\210721_L76_2P\1-003\Results\Cell_Compare.csv'
compare_list = []
with open(csv_path) as f:
    f_tsv = csv.reader(f, delimiter=',')
    headers = next(f_tsv)
    for row in f_tsv:
        compare_list.append(row)
#%% Last, add Run 03 data into origin.
new_cell_dic = {}
for i in range(len(compare_list)):
    c_cellname = 'L76_210721A_'+ot.Bit_Filler(i)
    tc = all_cells[c_cellname]
    targeted_03_cellid = int(compare_list[i][1])
    if targeted_03_cellid == -1:
        tc['In_Run']['Run003'] = False
    else:
        tc['In_Run']['Run003'] = True
        tc['Run003'] = {}
        tc['Run003']['F_train'] = all_03_F[targeted_03_cellid]
        tc['Run003']['dF_train'] = all_03_dF[targeted_03_cellid]
        tc['Run003']['Mean_F'] = all_03_F[targeted_03_cellid].mean()
        tc['Run003']['STD_F'] = all_03_F[targeted_03_cellid].std()
    new_cell_dic[c_cellname] = tc
ot.Save_Variable(r'K:\Test_Data\2P\210721_L76_2P', 'L76_210721A_All_Cell_Include_Run03', new_cell_dic,'.ac')

##########################################################################
'''Till Now, We get all cell tran in 210721 Data.'''


