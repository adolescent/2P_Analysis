# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 16:25:07 2020

@author: ZR
Test Run File
Do not save.
"""
#%%
import more_itertools as mit

test = list(mit.split_when(stim_train,lambda x, y: (x-y) >0))
#%% This part will count how many frame have already be read.
current_id = 2# remember to start from zero!
counter = 0 # count number of frames before current list.
current_list_series = 5
for i in range(current_list_series):
    counter = counter + len(test[i])
real_current_id = counter + current_id




#%% This part is real run areas.
import numpy as np
dF_Dic = {}
ignore_ISI_frame = 1
all_keys = list(F.keys())
cutted_stim_train = list(mit.split_when(stim_train,lambda x, y: (x-y) >0))
for i in range(len(all_keys)):
    current_cell_train = F[all_keys[i]]
    frame_counter = 0
    current_cell_dF_train = []
    for j in range(len(cutted_stim_train)):
        current_stim_train = np.asarray(cutted_stim_train[j])
        current_F_train = np.asarray(current_cell_train[frame_counter:(frame_counter+len(current_stim_train))])
        null_id = np.where(current_stim_train == -1)[0]
        null_id = null_id[ignore_ISI_frame:]
        current_base = current_F_train[null_id].mean()
        current_dF_train = (current_F_train-current_base)/current_base
        current_cell_dF_train.extend(current_dF_train)
        # Then add frame counter at last.
        frame_counter = frame_counter + len(cutted_stim_train[j])
    dF_Dic[i] = current_cell_dF_train
#%%



