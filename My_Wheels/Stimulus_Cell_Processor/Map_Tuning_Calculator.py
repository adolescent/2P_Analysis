# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 14:17:32 2021

@author: ZR
"""


import pandas as pd
import OS_Tools_Kit as ot
from Decorators import Timer



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
                c_weight = c_tuning*float(normed_input_cell_frame.loc[cc])
                cell_property_frame.loc[c_prop,cc] = c_weight
    # Till now, we get all all cell tuning property, then we will sum them together.
    used_cell_each_prop = cell_property_frame.count(1)
    map_tunings = cell_property_frame.sum(1)/used_cell_each_prop
    return map_tunings




@Timer
def PC_Tuning_Calculation(all_PCA_comp,day_folder):
    '''
    This function is used to calculate average tuning of specific graph.
    Can be used to graph which is different from global average.
    Parameters
    ----------
    all_PCA_comp : (pd Frame)
        All PCA components.
    day_folder : (str)
        Data save folder. tuning need to be calculated first.

    Returns
    -------
    PC_Tunings : TYPE
        DESCRIPTION.

    '''
    
    tuning_dic = ot.Load_Variable(day_folder,'All_Tuning_Property.tuning')
    all_PC_names = all_PCA_comp.columns.tolist()
    PC_Tunings = {}
    PC_Tuning_Matrix = pd.DataFrame()
    for i,c_pc in enumerate(all_PC_names):
        c_pc_comp = all_PCA_comp.loc[:,c_pc]
        normed_c_comp = c_pc_comp/abs(c_pc_comp).max()
        c_PC_tunings = Map_Tuning_Core(tuning_dic,normed_c_comp)
        PC_Tunings[c_pc] = (c_PC_tunings,c_PC_tunings.idxmax())# get tunings and max tuning graph
        PC_Tuning_Matrix.loc[:,c_pc] = c_PC_tunings
        
    return PC_Tunings,PC_Tuning_Matrix
    




