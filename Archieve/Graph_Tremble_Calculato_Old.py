# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 13:34:33 2020

@author: ZR
Calculate graph tremble. Return tramle plot.
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
        cut_shape = (10,5),
        boulder = 20,
        move_method = 'former',
        base = [],
        center_method = 'weight'
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
    move_method : ('average'or'former'or'input'), optional
        Method of bais calculation. The default is 'former'. 
        'average' bais use all average; 'former' bais use fomer frame; 'input' bais need to be given.
    base : (2D_NdArray), optional
        If move_method == 'input', base should be given here. The default is [].
    center_method : ('weight' or 'binary'), optional
        Method of center find. Whether we use weighted intense.The default is 'weight'.

    Returns
    -------
    mass_center_maps(Graph)
        A plotted graph, showing movement trace of mass center.
    tremble_plots : (List)
        List of all fracture graph tremble list.
    tremble_information : (Dic)
        Dictionary of tramble informations.
        Data type of tremble_information:
            keys:frame ID
            data are lists, every element in list indicate a fracture grpah, ID in cut graph.
            list elements are turples, each turple[0] are move vector, turple[1] as move distance.
            

    '''
    all_tif_name = OS_Tools.Get_File_Name(data_folder,graph_type)
    average_graph = Graph_Tools.Average_From_File(all_tif_name)
    tremble_information = {}
    # get base of align first.
    if move_method == 'average':
        base_graph = average_graph
    elif move_method == 'input':
        base_graph = base
    elif move_method == 'former':
        base_graph = cv2.imread(all_tif_name[0],-1)# Use first frame as base.
    
    # cycle all graph to generate tremble plots.
    for i in range(len(all_tif_name)):
        # Process input graph, get cell 
        current_graph = cv2.imread(all_tif_name[i],-1)
        processed_cell_graph = None
        #Cut Graph as described
        _,_,_,cutted_current_graph = Graph_Cutter(processed_cell_graph,boulder,cut_shape)
        _,_,_,cutted_base = Graph_Cutter(base_graph,boulder,cut_shape)
        # Renew base if former mode.
        if move_method == 'former':
            base_graph = cv2.imread(all_tif_name[i],-1)
        # Then cycle all cutted_fracture, to calculate movement of every fracture graph.
        current_frame_move_list = []
        for j in range(len(cutted_current_graph)):
            temp_graph_part = cutted_current_graph[j]
            temp_base_part = cutted_base[j]
            temp_graph_center,_ = Graph_Tools.Graph_Center_Calculator(temp_graph_part,center_mode = center_method)
            temp_base_center,_ = Graph_Tools.Graph_Center_Calculator(temp_base_part,center_mode = center_method)
            temp_tremble_vector,temp_tremble_dist = Calculator.Vector_Calculate(temp_base_center, temp_graph_center)
            current_frame_move_list.append((temp_tremble_vector,temp_tremble_dist))
        tremble_information[i] = current_frame_move_list
        
        
    # Then, plot mass center plots. This will show change of mass center position.
    if move_method == 'input':
        print('No Mass Center plot Generated.')
        mass_center_maps = False
    elif move_method == 'average':# If average, use current location
        mass_center_maps = []
        for i in range(len(tremble_information[0])):# Cycle all fracture
            fig = plt.figure()
            ax = plt.subplot()
            for j in range(len(tremble_information)):# Cycle all frame
                current_point = tremble_information[j][i][0]
                ax.scatter(current_point[1],current_point[0],alpha = 1,s = 5)
            mass_center_maps.append(fig)
            plt.close()
    elif move_method == 'former':
        mass_center_maps = []
        for i in range(len(tremble_information[0])):# Cycle all fracture
            fig = plt.figure()
            ax = plt.subplot()
            current_point = (0,0)
            for j in range(len(tremble_information)):# Cycle all frame
                current_point = (current_point[0]+tremble_information[j][i][0][0],current_point[1]+tremble_information[j][i][0][1])
                ax.scatter(current_point[1],current_point[0],alpha = 1,s = 5)
            mass_center_maps.append(fig)
            plt.close()
            
            
    # At last, plot tremble dist plots. Each fracture have a plot.
    tremble_plots = {}
    for i in range(len(tremble_information[0])):# Cycle all fractures
        current_tremble_plot = []
        for j in range(len(tremble_information)): # Cycle all frame
            current_dist = tremble_information[j][i][1]
            current_tremble_plot.append(current_dist)
        tremble_plots[i] = np.asarray(current_tremble_plot)
    return mass_center_maps,tremble_plots,tremble_information