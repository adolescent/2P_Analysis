# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 12:16:01 2021

@author: ZR
"""


import Filters
import numpy as np
import pandas as pd

def Spike_Count(data_Frame,window = 5,win_step = 1,
                fps = 1.301,pass_band = (0.05,0.5)):
    '''
    Count total dF/F in a time window, to evaluate cell activity.

    Parameters
    ----------
    data_Frame : (pd Frame)
        Data frame of cell data. One row a cell.
    window : (int), optional
        Average window size in seconds. The default is 5.
    win_step : (int), optional
        window step in seconds. The default is 1.
    fps : (float), optional
        Capture frequency. The default is 1.3.
    pass_band : TYPE, optional
        DESCRIPTION. The default is (0.05,0.5).

    Returns
    -------
    activation_Diagram : TYPE
        DESCRIPTION.

    '''
    acn = data_Frame._stat_axis.values.tolist()# get all cell name
    HP_Para = pass_band[0]*2/fps
    LP_Para = pass_band[1]*2/fps
    dF_F_Frame = data_Frame.copy()# deeph copy
    for i in range(len(acn)):
        c_train = data_Frame.loc[acn[i]]
        c_filted_train = Filters.Signal_Filter(c_train,filter_para = (HP_Para,LP_Para))
        frame_num = len(c_filted_train)
        most_unactive = np.array(sorted(c_filted_train)[0:int(frame_num*0.1)])# Use least active 10% as base
        c_base = most_unactive.mean()
        c_dF_F_train = np.clip((c_filted_train-c_base)/c_base,0,None)
        dF_F_Frame.loc[acn[i]] = c_dF_F_train
    # After generating dF/F trains, we will calcuate
    window_length = int(window*fps)
    window_step = int(win_step*fps)
    win_num = 1+int((frame_num-window_length)/window_step)# Down stair round
    spike_counter = pd.DataFrame(index =acn)
    for i in range(win_num):
        c_win = dF_F_Frame.iloc[:,window_step*i:(window_step*i+window_length)]
        c_count = c_win.sum(1)# sum all dF/F values.
        spike_counter[i] = c_count
    # 加一个Z分数的功能
    cell_avr = spike_counter.mean(1)
    cell_std = spike_counter.std(1)
    Z_counter = (spike_counter.sub(cell_avr,axis = 0)).div(cell_std,axis = 0)
    Z_counter = Z_counter.clip(-5,5,axis = 0)
    return spike_counter,Z_counter


def Pre_Processed_Data_Count(input_frame,window = 5,win_step = 1,fps = 1.301):
    
    acn = input_frame._stat_axis.values.tolist()# get all cell name
    frame_num = input_frame.shape[1]
    window_length = int(window*fps)
    window_step = int(win_step*fps)
    win_num = 1+int((frame_num-window_length)/window_step)# Down stair round
    spike_counter = pd.DataFrame(index =acn)
    for i in range(win_num):
        c_win = input_frame.iloc[:,window_step*i:(window_step*i+window_length)]
        c_count = c_win.sum(1)# sum all dF/F values.
        spike_counter[i] = c_count
    # 加一个Z分数的功能
    cell_avr = spike_counter.mean(1)
    cell_std = spike_counter.std(1)
    Z_counter = (spike_counter.sub(cell_avr,axis = 0)).div(cell_std,axis = 0)
    Z_counter = Z_counter.clip(-5,5,axis = 0)
    return spike_counter,Z_counter
