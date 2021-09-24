# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 14:17:32 2021

@author: ZR
"""


import pandas as pd


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
    cell_property_frame = pd.DataFrame(index=all_property_names)
    # Calculate tuning dics
    normed_input_cell_frame = input_cell_frame/abs(input_cell_frame).max()
    
    for i,cc in enumerate(all_cell_in_graph):
        c_D_value = tuning_dic[cc]
        for j,c_prop in enumerate(all_property_names):
            if c_prop in c_D_value:
                c_tuning = c_D_value[c_prop]['Cohen_D']
                c_weight = c_tuning*normed_input_cell_frame.loc[cc]
                cell_property_frame.loc[c_prop,cc] = c_weight
    # Till now, we get all all cell tuning property, then we will sum them together.
    used_cell_each_prop = cell_property_frame.count(1)
    map_tunings = cell_property_frame.sum(1)/used_cell_each_prop
    return map_tunings






