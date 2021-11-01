# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 13:38:13 2021

@author: ZR

Get Cross Correlation between maps and map series.

"""

import scipy.stats as stats


def Correlation_Core(map_A,map_B):
    common_cells = list(set(map_A.index)&set(map_B.index))
    common_cells.sort()
    used_A_map = map_A.loc[common_cells]
    used_B_map = map_B.loc[common_cells]
    pearson_r,p_value = stats.pearsonr(used_A_map,used_B_map)
    return pearson_r,p_value
    


def PC_Comp_vs_t_Maps(PC_components,all_t_graphs):
    
    return 


    