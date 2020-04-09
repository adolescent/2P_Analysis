# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 13:40:53 2020

@author: zhang
"""
#%% Function 1： Cell Location Compare.
import numpy as np

def Cell_Location_Compare(
        Cell_Set_A,
        Cell_Set_B,
        shift_limit = 10
        ):
    """
    Compare similarity of 2 cell sets

    Parameters
    ----------
    Cell_Set_A : (skimage region list)
        Base sets. Use A to compare B.
    Cell_Set_B : (skimage region list)
        Reference sets.
    shift_limit : (int)
        Max distance of match. If all pair distance bigger than it, return no match.

    Returns
    -------
    Compare_Dictionary : (Dic)
        Match result dictionary. Keys are cell a ids,

    """
    # Get center ids.
    A_Centers = []
    B_Centers = []
    for i in range(len(Cell_Set_A)):
        A_Centers.append(Cell_Set_A[i].centroid)
    for i in range(len(Cell_Set_B)):
        B_Centers.append(Cell_Set_B[i].centroid)
    # Then calculate Compare situation.Use A set as base.
    Compare_Dictionary = {}
    for i in range(len(A_Centers)):# for every A cell
        temp_dist = np.zeros(len(B_Centers))
        y_base,x_base = A_Centers[i]
        # calculate all dists.
        for j in range(len(B_Centers)):
            y_temp,x_temp = B_Centers[j]
            dist_sqr = (y_base-y_temp)**2+(x_base-x_temp)**2
            temp_dist[j] = dist_sqr # This is all B distant with A[i]
        # write least dist and matched B ids.
        current_least_dist = temp_dist.min()
        if current_least_dist < shift_limit**2:
            current_match_id = np.where(temp_dist == current_least_dist)[0][0]
            middle_point = ((y_base+B_Centers[current_match_id][0])/2,(x_base+B_Centers[current_match_id][1])/2)
        else:# If not match
            current_match_id = -1
            middle_point  = (0,0)
        Compare_Dictionary[i] = (current_match_id,middle_point,current_least_dist)
    return Compare_Dictionary
