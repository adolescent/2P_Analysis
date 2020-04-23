# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 16:25:07 2020

@author: ZR
Test Run File
Do not save.
"""
#%%
import more_itertools as mit
import numpy as np
import My_Wheels.List_Operation_Kit as List_Tools
dF_F_trains = {}
#%%
stim_train = np.asarray(stim_train)
blank_location = np.where(stim_train == 0)[0]
cutted_blank_location = list(mit.split_when(blank_location,lambda x,y:(y-x)>1))
all_blank_start_frame = [] # This is the start frame of every blank.
for i in range(len(cutted_blank_location)):
    all_blank_start_frame.append(cutted_blank_location[i][0])
    
#%% Get base_F_of every blank.
all_keys = list(F_value_Dictionary.keys())
for i in range(len(all_keys)):
    current_key = all_keys[i]
    current_cell_F_train = F_value_Dictionary[current_key]
    # First, get base F of every blank.
    all_blank_base_F = [] # base F of every blank.
    for j in range(len(cutted_blank_location)):
        all_blank_base_F.append(current_cell_F_train[cutted_blank_location[j]].mean())
    # Then, generate dF train.
    current_dF_train = []
    for j in range(len(current_cell_F_train)):
        current_F = current_cell_F_train[j]
        _,current_base_loc = List_Tools.Find_Nearest(all_blank_start_frame,j)
        current_base = all_blank_base_F[current_base_loc]
        current_dF_F = (current_F-current_base)/current_base
        current_dF_train.append(current_dF_F)
    dF_F_trains[all_keys[i]] = np.asarray(current_dF_train)