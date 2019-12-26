# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 15:47:51 2019

@author: ZR

Graph Operation kits, this tool box aims at doing all graph Works
"""

import cv2
import numpy as np

#%% Function1: Graph Average(From File).

def Average_From_File(Name_List):
    """
    Average Graph Files, return an aligned matrix. RGB Graph shall be able to use it (not tested).

    Parameters
    ----------
    Name_List : (list)
        File Name List. all list units shall be a direct file path.

    Returns
    -------
    averaged_graph : (2D ndarray, float64)
        Return averaged graph, data type f8 for convenient.

    """
    graph_num = len(Name_List)
    temple_graph = cv2.imread(Name_List[0],-1)
    averaged_graph = np.zeros(shape = temple_graph.shape,dtype = 'f8')
    for i in range(graph_num):
        current_graph = cv2.imread(Name_List[i],-1).astype('f8')# Read in graph as origin depth, and change into f8
        averaged_graph += current_graph/graph_num
    return averaged_graph

#%% Function2: Clip And Normalize input graph
def Clip_And_Normalize(input_graph,clip_std = 2.5,normalization = True,bit = 'u2'):
    """
    Clip input graph,then normalize them to specific bit depth, output graph be shown directly.
    
    Parameters
    ----------
    input_graph : (2D ndarray)
        Input graph matrix. Need to be a 2D ndarray.
    clip_std : (float), optional
        How much std will the input graph be clipped into. The default is 2.5, holding 99% data unchanged.
        This Variable can be set to -1 to skip clip.
    normalization : (Bool), optional
        Whether normalization is done here. The default is True, if False, no normalization will be done here.
    bit : (str), optional
        dtype of output graph. This parameter will affect normalization width. The default is 'u2'.

    Returns	
    -------
    processed_graph : (2D ndarray)
        Output graphs.

    """
    #Step1, clip
    input_graph = input_graph.astype('f8')
    
    
    return processed_graph