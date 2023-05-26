# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 14:26:58 2022

@author: ZR
"""

import Graph_Operation_Kit as gt
import List_Operation_Kit as lt
import OS_Tools_Kit as ot

day_folder = r'G:\Test_Data\2P\220223_L85_2P_Check'
sub_folders = ['1-001','1-002','1-003','1-004','1-005','1-006','1-007','1-008']
data_folders = lt.List_Annex([day_folder], sub_folders)
for i,c_folder in enumerate(data_folders):
    all_tif_name = ot.Get_File_Name(c_folder)
    c_graph = gt.Average_From_File(all_tif_name)
    clipped_graph = gt.Clip_And_Normalize(c_graph,5)
    result_folder = c_folder+r'\Results'
    ot.mkdir(result_folder)
    gt.Show_Graph(clipped_graph, 'Average_Graph', result_folder)