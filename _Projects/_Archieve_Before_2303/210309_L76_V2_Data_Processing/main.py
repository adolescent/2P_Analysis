# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 10:15:43 2021

@author: ZR
"""

import My_Wheels.List_Operation_Kit as lt
from My_Wheels.Translation_Align_Function import Translation_Alignment
from My_Wheels.Standard_Parameters.Sub_Graph_Dics import Sub_Dic_Generator

day_folder = [r'K:\Test_Data\2P\210309_L76_2P']
run_folder = ['1-001',
              '1-002',
              '1-003',
              '1-004',
              '1-005',
              '1-006',
              '1-007',
              '1-008',
              '1-009',
              '1-010',
              '1-011',
              '1-012',
              '1-013',
              '1-014',
              '1-015',
              '1-016'
              ]
all_run_folders = lt.List_Annex(day_folder, run_folder)
#%% Align all graphs automatically. Based on global average.
Translation_Alignment(all_run_folders,big_memory_mode=True)
#%% Then, generate all stim maps.
from My_Wheels.Stim_Frame_Align import One_Key_Stim_Align
One_Key_Stim_Align(r'K:\Test_Data\2P\210309_L76_2P\210308_L76_2P_stimuli')
#%% Move Stim Frame Align file into folders, then we can generate stimulus maps.
G16_Para = Sub_Dic_Generator('G16_2P')
from My_Wheels.Standard_Stim_Processor import One_Key_Frame_Graphs
One_Key_Frame_Graphs(r'K:\Test_Data\2P\210309_L76_2P\1-005', G16_Para)
RFSize_Para = Sub_Dic_Generator('RFSize',{'Size':[1,1.5,2,2.5,4],'Dir':[270,315,0,45,90,135,180,225]})
One_Key_Frame_Graphs(r'K:\Test_Data\2P\210309_L76_2P\1-013',RFSize_Para)
SFTF_Para = Sub_Dic_Generator('SFTF',{'SF':[0.5,0.75,1,1.25,1.5],'TF':[2,4,8],'Dir':[315,45,135,225]})
One_Key_Frame_Graphs(r'K:\Test_Data\2P\210309_L76_2P\1-014',SFTF_Para)
S3D8_Para = Sub_Dic_Generator('Shape3Dir8')
One_Key_Frame_Graphs(r'K:\Test_Data\2P\210309_L76_2P\1-015',S3D8_Para)
C7D8_Para = Sub_Dic_Generator('Color7Dir8+90')
One_Key_Frame_Graphs(r'K:\Test_Data\2P\210309_L76_2P\1-016',C7D8_Para)
