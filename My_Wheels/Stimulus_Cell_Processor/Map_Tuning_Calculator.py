# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 14:17:32 2021

@author: ZR
"""

import OS_Tools_Kit as ot
import numpy as np


def Map_Tuning_Core(tuning_dic,input_cell_frame):
    
    '''
    Get tuning scores of a single map.

    Parameters
    ----------
    tuning_dic : (dic)
        Dictionary of Cell tuning property. Use '.tuning' file
    input_cell_frame : (pd Frame)
        Cell combinations. Usually a map or a PC component. Need to be pd frame.

    Returns
    -------
    map_tuning_score : (pd Frame)
        Frame of all tuinng scores of given graph.

    '''
    # Get all tuning names.
    all_cell_in_graph = input_cell_frame.index.tolist()
    full_tuning_cell = all_cell_in_graph[0]
    for i,ccn in enumerate(all_cell_in_graph):
        if len(tuning_dic[ccn].keys()) > len(tuning_dic[full_tuning_cell].keys()):
            full_tuning_cell = ccn
    all_property_names = list(tuning_dic[full_tuning_cell].keys())
    all_property_names.sort(reverse = True)
    all_property_names = all_property_names[5:]
    # Calculate tuning dics
    
    
    normed_input_cell_frame = input_cell_frame/abs(input_cell_frame).max()
    for i,cc in enumerate(all_cell_in_graph):
        c_D_value = tuning_dic[cc]
        
        
    return map_tuning_scores






