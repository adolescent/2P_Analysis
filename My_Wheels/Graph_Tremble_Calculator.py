# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 14:47:46 2020

@author: ZR
"""

import My_Wheels.OS_Tools_Kit as OS_Tools
import My_Wheels.Graph_Operation_Kit as Graph_Tools
from My_Wheels.Graph_Cutter import Graph_Cutter
import My_Wheels.Calculation_Functions as Calculator
import cv2
from matplotlib import pyplot as plt 
import numpy as np

def Tremble_Calculator_From_File(
        data_folder,
        graph_type = '.tif',
        cut_shape = (8,8),
        boulder = 20,
        base_method = 'former',
        base = [],
        ):
    '''
    Calculate align tremble from graph. This program is used to evaluate align quality.
    
    Parameters
    ----------
    data_folder : (str)
        Data folder of graphs.
    graph_type : (str),optional
        Extend name of input grahp. The default is '.tif'.
    cut_shape : (turple), optional
        Shape of fracture cut. Proper cut will . The default is (10,5).
    boulder : (int),optional
        Boulder of graph. Cut and not used in following calculation.The default is 20.        
    base_method : ('average'or'former'or'input'), optional
        Method of bais calculation. The default is 'former'. 
        'average' bais use all average; 'former' bais use fomer frame; 'input' bais need to be given.
    base : (2D_NdArray), optional
        If move_method == 'input', base should be given here. The default is [].

    Returns
    -------
    mass_center_maps(Graph)
        A plotted graph, showing movement trace of mass center.
    tremble_plots : (List)
        List of all fracture graph tremble list.
    tremble_information : (Dic)
        Dictionary of tramble informations.
        Data type of tremble_information:
    '''
    all_tif_name = OS_Tools.Get_File_Name(data_folder,file_type = graph_type)
    average_graph = Graph_Tools.Average_From_File(all_tif_name)
    tremble_information = {}
    #1. Get base graph first.
    if base_method == 'input':
        base_graph = base
    elif base_method == 'average':
        base_graph = average_graph
    elif base_method == 'former':
        base_graph = cv2.imread(all_tif_name[0],-1)# First input graph.
    else:
        raise IOError('Invalid Base Method, check please.\n')
        
    
        
        