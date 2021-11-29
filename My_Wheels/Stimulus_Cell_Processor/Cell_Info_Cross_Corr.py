# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 13:38:13 2021

@author: ZR

Get Cross Correlation between maps and map series.

"""

import scipy.stats as stats
import pandas as pd


def Correlation_Core(map_A,map_B):
    '''
    Pearson R of 2 graphs.

    Parameters
    ----------
    map_A : pd Frame
        A map. Need to be a set of cells
    map_B : pd Frame
        B map. Need to be a set of cells

    Returns
    -------
    pearson_r : (float)
        Pearson R value of the given 2 graph.
    p_value : (float)
        P value of r test.

    '''
    common_cells = list(set(map_A.index)&set(map_B.index))
    common_cells.sort()
    used_A_map = map_A.loc[common_cells]
    used_B_map = map_B.loc[common_cells]
    pearson_r,p_value = stats.pearsonr(used_A_map,used_B_map)
    return pearson_r,p_value
    


def PC_Comp_vs_t_Maps(PC_components,all_t_graphs,p_thres = 0.05):
    
    all_stim_names =list(all_t_graphs.keys())
    all_PC_names = PC_components.columns.tolist()
    PC_stim_Frames = pd.DataFrame()
    PC_stim_p = pd.DataFrame()
    for i,c_stim in enumerate(all_stim_names):
        c_stim_map = all_t_graphs[c_stim].loc['CohenD']
        for j,c_PC in enumerate(all_PC_names):
            c_PC_map = PC_components.loc[:,c_PC]
            PC_stim_Frames.loc[c_stim,c_PC],PC_stim_p.loc[c_stim,c_PC] = Correlation_Core(c_stim_map, c_PC_map)
    
    sig_PC_stim_Frames = PC_stim_Frames*(PC_stim_p<p_thres)
    PC_judge  = {}
    for i,c_PC in enumerate(all_PC_names):
        c_series = sig_PC_stim_Frames.loc[:,c_PC]
        PC_judge[c_PC] = (c_series,abs(c_series).idxmax())
    
    return PC_stim_Frames,PC_stim_p,PC_judge


    