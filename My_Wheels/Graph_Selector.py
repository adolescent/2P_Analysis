# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 16:27:37 2020

@author: ZR

This part contains mulitple functions that can select specific IDs and their averages.

"""
import My_Wheels.Graph_Operation_Kit as Graph_Tools
import My_Wheels.OS_Tools_Kit as OS_Tools
import numpy as np
import cv2
#%% Function 1, Partial Averages.

def Partial_Average_From_File(data_folder,
                              start_frame,
                              stop_frame,
                              graph_type = '.tif',
                              LP_Para = False,HP_Para = False,filter_method = False):
    all_tif_name = np.array(OS_Tools.Get_File_Name(data_folder,file_type = graph_type))
    used_tif_name = all_tif_name[start_frame:stop_frame]
    Averaged_Graph = Graph_Tools.Average_From_File(used_tif_name,LP_Para,HP_Para,filter_method)
    return Averaged_Graph
#%% Function2, Calculate intensity list of input graphs.
