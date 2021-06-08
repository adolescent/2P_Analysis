# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 15:06:18 2021

@author: ZR
"""
import Graph_Operation_Kit as gt
import OS_Tools_Kit as ot
import numpy as np
from Cell_Processor import Cell_Processor

work_path = r'D:\ZR\_MyCodes\2P_Analysis\_Projects\210616_Annual_Report'
#%% Graph1, generate average graph of different run.
# Use G8 response as graph base.
graph_names_0604 = ot.Get_File_Name(r'K:\Test_Data\2P\210604_L76_2P\1-016\Results\Final_Aligned_Frames')
avr_0604 = gt.Average_From_File(graph_names_0604)
graph_names_0123 = ot.Get_File_Name(r'K:\Test_Data\2P\210123_L76_2P\1-011\Results\Aligned_Frames')
avr_0123 = gt.Average_From_File(graph_names_0123)
clipped_0604 = np.clip((avr_0604.astype('f8'))*30,0,65535).astype('u2')
clipped_0123 = np.clip((avr_0123.astype('f8'))*30,0,65535).astype('u2')
gt.Show_Graph(clipped_0604, 'Average_0604',work_path)
gt.Show_Graph(clipped_0123, 'Average_0123',work_path)
#%% Graph2, get cell layout of 210401 and 210413
CP_0401 = Cell_Processor(r'K:\Test_Data\2P\210401_L76_2P')
CP_0413 = Cell_Processor(r'K:\Test_Data\2P\210413_L76_2P')
all_cell_name_0401 = CP_0401.all_cell_names
all_cell_name_0413 = CP_0413.all_cell_names
from Cross_Day_Cell_Layout import Cross_Day_Cell_Layout
h = Cross_Day_Cell_Layout(r'K:\Test_Data\2P\210401_L76_2P',r'K:\Test_Data\2P\210413_L76_2P', all_cell_name_0401, all_cell_name_0413)
#%% Analyze spon series of 0413.
from Spontaneous_Processor import Spontaneous_Processor
SP = Spontaneous_Processor(r'K:\Test_Data\2P\210413_L76_2P',spon_run = 'Run001')
pc_dic_0_10 = SP.Do_PCA(0,600)
pc_dic_10_20 = SP.Do_PCA(600,1200)
pc_dic_20_30 = SP.Do_PCA(1200,1800)
pc_dic_30_40 = SP.Do_PCA(1800,2400)
pc_dic_0_45_all = SP.Do_PCA(0,2770)
pc_dic_stim_10 = SP.Do_PCA(2800,3400)
test_SP = Spontaneous_Processor(r'K:\Test_Data\2P\210413_L76_2P',spon_run = 'Run007')
test_PCA = test_SP.Do_PCA()
after_SP = Spontaneous_Processor(r'K:\Test_Data\2P\210413_L76_2P',spon_run = 'Run009')
after_pc_all = after_SP.Do_PCA()
after_pc_0_10 = after_SP.Do_PCA(0,600)
after_pc_50_60 = after_SP.Do_PCA(3000,3600)

