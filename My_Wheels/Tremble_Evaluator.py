# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 10:54:45 2020

@author: ZR
This file used to Evaluate align quality. Through mass center 

"""
from My_Wheels.Graph_Cutter import Graph_Cutter
import My_Wheels.OS_Tools_Kit as OS_Tools
import My_Wheels.Graph_Operation_Kit as Graph_Tools

def Tremble_Evaluator(
        data_folder,
        ftype = '.tif',
        boulder_ignore = 20,
        cut_shape = (4,4)
        ):
    
    all_tif_name = OS_Tools.Get_File_Name(data_folder,file_type = ftype)
    average_graph = Graph_Tools.Average_From_File(all_tif_name)
    schametic_graph,graph_location_list,after_size,cutted_graph_dics = Graph_Cutter(average_graph,boulder_ignore,cut_shape)
    for i in range(len(all_tif_name)):
        Graph_Cutter()
    return Tremble_Dic