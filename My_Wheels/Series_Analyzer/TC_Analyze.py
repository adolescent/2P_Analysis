# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 12:01:06 2021

@author: ZR

Functions here is used to process time course information.

"""
import numpy as np
import pandas as pd


def Peak_Counter(component_series,thres_std = 1,
                   win_size = 180,win_step = 60,fps = 1.301):
    '''
    Count signal repeat in time windows.

    Parameters
    ----------
    component_series : (pd Series/nd array)
        Time course train.
    thres_std : (float), optional
        Std of threshold. Activation above thres is a repeat. The default is 1.
    win_size : (int), optional
        Count window (second). The default is 180.
    win_step : (int), optional
        Count window step (second). The default is 60.
    fps : (float), optional
        Capture frequency. The default is 1.301.

    Returns
    -------
    active_counter : (nd array)
        Array of active counter.
    '''
    # Initialize
    arrayed_input_data = np.array(component_series)
    frame_num = len(arrayed_input_data)
    thres = arrayed_input_data.mean()+arrayed_input_data.std()*thres_std
    neg_thres = arrayed_input_data.mean()-arrayed_input_data.std()*thres_std
    
    # calculate frame by frame.
    win_size_frame = int(win_size*fps)
    win_step_frame = int(win_step*fps)
    window_num = 1+int((frame_num-win_size_frame)/win_step_frame)
    peak_counter = np.zeros(window_num)
    for i in range(0,window_num):
        c_window = arrayed_input_data[win_step_frame*i:win_step_frame*i+win_size_frame]
        current_peak_num = (c_window>thres).sum()+(c_window<neg_thres).sum()
        peak_counter[i] = current_peak_num
    
    return peak_counter

