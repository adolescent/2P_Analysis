# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 10:54:45 2020

@author: ZR
This file used to Evaluate align quality. Through mass center 

"""
from My_Wheels.Graph_Cutter import Graph_Cutter
import My_Wheels.OS_Tools_Kit as OS_Tools
import My_Wheels.Graph_Operation_Kit as Graph_Tools
import cv2

def Tremble_Evaluator(
        data_folder,
        ftype = '.tif',
        binary_thres = 1,
        boulder_ignore = 20,
        cut_shape = (4,4)
        ):
    '''
    Evaluate tremble for graphs in given folder.

    Parameters
    ----------
    data_folder : str
        graph folder.
    ftype : str, optional
        Dtype of graphs. The default is '.tif'.
    boulder_ignore : int, optional
        Out part will be ignored. The default is 20.
    cut_shape : TYPE, optional
        Shape of graph cut. The default is (4,4).

    Returns
    -------
    center_move_dic : TYPE
        DESCRIPTION.
    center_loc_dic : TYPE
        DESCRIPTION.

    '''
    # Initialization
    save_folder = data_folder+r'\Tremble_Evaluation'
    OS_Tools.mkdir(save_folder)
    all_tif_name = OS_Tools.Get_File_Name(data_folder,file_type = ftype)
    average_graph = Graph_Tools.Average_From_File(all_tif_name)
    schametic_graph,_,_,_ = Graph_Cutter(average_graph,boulder_ignore,cut_shape)
    Graph_Tools.Show_Graph(schametic_graph, 'Schametic_Graph', save_folder)
    # then calculate center movement and center location.
    center_move_dic = {}
    centere_loc_dic = {}
    for i in range(len(all_tif_name)):
        current_graph = cv2.imread(all_tif_name[i],-1)
        
        _,_,_,current_cutted_graphs = Graph_Cutter(current_graph,boulder_ignore,cut_shape)
        
        
    return center_move_dic,center_loc_dic