# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 12:17:34 2021

@author: ZR
This function is used to generate stim t maps auto correlation matrix.
"""

import pandas as pd
from Stimulus_Cell_Processor.Cell_Info_Cross_Corr import Correlation_Core


def T_Graph_AutoCorr(All_t_Graphs):
    
    all_graph_name = list(All_t_Graphs.keys())
    Corr_Matrix = pd.DataFrame()
    p_Matrix = pd.DataFrame()
    for i,A_map in enumerate(all_graph_name):
        for j,B_map in enumerate(all_graph_name):
            set_A = All_t_Graphs[A_map].loc['CohenD']
            set_B = All_t_Graphs[B_map].loc['CohenD']
            c_r,c_p = Correlation_Core(set_A, set_B)
            Corr_Matrix.loc[A_map,B_map] = c_r
            p_Matrix.loc[A_map,B_map] = c_p
    return Corr_Matrix,p_Matrix


def PC_Comps_AutoCorr(used_PC_comps):
    Corr_Matrix = pd.DataFrame()
    p_Matrix = pd.DataFrame()
    used_all_PC_name = used_PC_comps.columns.tolist()
    for i,A_name in enumerate(used_all_PC_name):
        for j,B_name in enumerate(used_all_PC_name):
            set_A = used_PC_comps.loc[:,A_name]
            set_B = used_PC_comps.loc[:,B_name]
            c_r,c_p = Correlation_Core(set_A, set_B)
            Corr_Matrix.loc[A_name,B_name] = c_r
            p_Matrix.loc[A_name,B_name] = c_p
    return Corr_Matrix,p_Matrix
    