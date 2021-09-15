# -*- coding: utf-8 -*-
"""
Created on Tue May 18 15:28:34 2021

@author: ZR
These functions are used to do statistic calculations.

"""
from scipy.stats import ttest_rel,ttest_ind
import random
import numpy as np

def T_Test_Pair(A_set,B_set):
    sample_size = min(len(A_set),len(B_set))
    selected_A = random.sample(list(A_set),sample_size)
    selected_B = random.sample(list(B_set),sample_size)
    t_value,p_value = ttest_rel(selected_A,selected_B)
    if np.isnan(t_value):
        t_value = 0.0
        p_value = 1
    CohenD = t_value/np.sqrt(sample_size)
    return t_value,p_value,CohenD

def T_Test_Ind(A_set,B_set):
    sample_size = min(len(A_set),len(B_set))
    selected_A = random.sample(list(A_set),sample_size)
    selected_B = random.sample(list(B_set),sample_size)
    t_value,p_value = ttest_ind(selected_A,selected_B)
    if np.isnan(t_value):
        t_value = 0.0
        p_value = 1
    CohenD = t_value/np.sqrt(sample_size)
    return t_value,p_value,CohenD