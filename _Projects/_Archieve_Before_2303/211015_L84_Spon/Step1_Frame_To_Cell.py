# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 12:09:30 2021

@author: ZR
"""

# =============================================================================
# from Standard_Aligner import Standard_Aligner
# day_folder = r'G:\Test_Data\2P\211015_L84_2P'
# Sa = Standard_Aligner(day_folder, [1,2,3,4,5,6,7,8])
# Sa.One_Key_Aligner()
# 
# =============================================================================
# As afffine align seems to make more mistakes in these not so good area, use translation align only.
from Translation_Align_Function import Translation_Alignment
import List_Operation_Kit as lt

day_folder = r'G:\Test_Data\2P\211015_L84_2P'
all_folders = lt.List_Annex([day_folder],['1-001','1-002','1-003','1-004','1-005','1-006','1-007','1-008'])
Translation_Alignment(all_folders)



from Stim_Frame_Align import One_Key_Stim_Align
One_Key_Stim_Align(r'G:\Test_Data\2P\211015_L84_2P\211015_L84_stimuli')
from Standard_Stim_Processor import One_Key_Frame_Graphs
from Standard_Parameters.Sub_Graph_Dics import Sub_Dic_Generator
G16_Para = Sub_Dic_Generator('G16_2P')
One_Key_Frame_Graphs(r'G:\Test_Data\2P\211015_L84_2P\1-002', G16_Para)
OD_Para = Sub_Dic_Generator('OD_2P')
One_Key_Frame_Graphs(r'G:\Test_Data\2P\211015_L84_2P\1-006', OD_Para)
One_Key_Frame_Graphs(r'G:\Test_Data\2P\211015_L84_2P\1-008', OD_Para)
Hue_Para = Sub_Dic_Generator('HueNOrien4',para = 'Default')
One_Key_Frame_Graphs(r'G:\Test_Data\2P\211015_L84_2P\1-007', Hue_Para)

#%% Get manual cell
from Cell_Find_From_Graph import Cell_Find_From_Mannual
cell_dic = Cell_Find_From_Mannual(r'G:\Test_Data\2P\211015_L84_2P\_Manual_Cell\Cell_Mask.png',
                                  average_graph_path=r'G:\Test_Data\2P\211015_L84_2P\_Manual_Cell\Global_Average.tif',
                                  boulder = 5)
from Standard_Cell_Generator import Standard_Cell_Generator
Scg = Standard_Cell_Generator('L84', '211015', r'G:\Test_Data\2P\211015_L84_2P', [1,2,3,4,5,6,7,8])
Scg.Generate_Cells()
#%% Get tunings
from Stimulus_Cell_Processor.Tuning_Property_Calculator import Tuning_Property_Calculator
tuning_dic,tuning_checklists = Tuning_Property_Calculator(r'G:\Test_Data\2P\211015_L84_2P',
                                                          Orien_para=('Run002','G16_2P'),
                                                          OD_para = ('Run008','OD_2P'),                                                          
                                                          )

