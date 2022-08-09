# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 13:11:09 2022

@author: ZR
"""


def Win_Corr_Select(pc_info_criteria,pc_win):
    
    used_groups = pc_info_criteria[pc_info_criteria==True].index
    selected_pcs = pc_win.loc[used_groups,:]

    return selected_pcs