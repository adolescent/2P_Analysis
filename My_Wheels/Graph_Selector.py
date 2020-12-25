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
    '''
    Average specific part of graphs in the folder.

    Parameters
    ----------
    data_folder : (str)
        Data folder.
    start_frame : (int)
        Start ID of frame selection.
    stop_frame : (int)
        Stop ID of frame selection.
    graph_type : (str), optional
        Frame dtype. The default is '.tif'.
    LP_Para\HP_Para\filter_method : optional
        Filter parameters. The default is False.

    Returns
    -------
    Averaged_Graph : (2D Array)
        Averaged graphs.

    '''
    all_tif_name = np.array(OS_Tools.Get_File_Name(data_folder,file_type = graph_type))
    used_tif_name = all_tif_name[start_frame:stop_frame]
    Averaged_Graph = Graph_Tools.Average_From_File(used_tif_name,LP_Para,HP_Para,filter_method)
    return Averaged_Graph
#%% Function2, Calculate intensity list of input graphs.
def Intensity_Selector(data_folder,
                       graph_type = '.tif',
                       mode = 'biggest',
                       propotion = 0.05,
                       list_write = True
                       ):
    '''
    Select frames have biggest or smallest a.i., and generate average graphs.

    Parameters
    ----------
    data_folder : (str)
        Data folder.
    graph_type : (str), optional
        Data type of . The default is '.tif'.
    mode : ('biggest' or 'smallest'), optional
        Type of frame selection. The default is 'biggest'.
    propotion : (float), optional
        Propotion of graph selection. The default is 0.05.
    list_write : (bool), optional
        Whether we write down graph intensity data. The default is True.

    Returns
    -------
    averaged_graph : (2D Array)
        Averaged graph of selected frames.
    selected_graph_name : (ND List)
        List of selected graph names.

    '''
    all_graph_name = np.array(OS_Tools.Get_File_Name(data_folder,file_type = graph_type))
    graph_Num = len(all_graph_name)
    bright_data = np.zeros(graph_Num,dtype = 'f8')
    for i in range(graph_Num):
        current_graph = cv2.imread(all_graph_name[i],-1)
        bright_data[i] = np.mean(current_graph)
        # write bright data if required.
    if list_write == True:
        OS_Tools.Save_Variable(data_folder, 'brightness_info', bright_data)
    # Then select given mode frames.
    used_graph_num = int(graph_Num*propotion)
    if mode == 'biggest':
        used_graph_id = np.argpartition(bright_data,-used_graph_num)[-used_graph_num:]
    elif mode == 'smallest':
        used_graph_id = np.argpartition(bright_data,used_graph_num)[0:used_graph_num]
    selected_graph_name = all_graph_name[used_graph_id]
    averaged_graph = Graph_Tools.Average_From_File(selected_graph_name)
    return averaged_graph,selected_graph_name