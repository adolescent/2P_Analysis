# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 12:20:19 2022

@author: ZR

This function is used to cut series into small time windows.
"""
import numpy as np


def Generate_Windowed_Series(input_series,win_size = 300,win_step = 60,fps = 1.301):
    '''
    Generate windowed series.

    Parameters
    ----------
    input_series : (ND array)
        Input series need to be cut.
    win_size : (int), optional
        Size of each window.(Seconds) The default is 300.
    win_step : (int), optional
        Step of each window.(Seconds) The default is 60.
    fps : (float), optional
        Capturing rate. The default is 1.301.

    Returns
    -------
    windowed_series : (ND Array)
        Series of windowed graph.
    '''
    frame_num = len(input_series)
    win_frame = int(win_size*fps)
    step_frame = int(win_step*fps)
    win_num = (frame_num-win_frame)//step_frame+1 # Ignore last one.
    windowed_series = np.zeros(shape = (win_frame,win_num),dtype = 'f8')
    for i in range(win_num):
        c_window = input_series[i*step_frame:i*step_frame+win_frame]
        windowed_series[:,i] = c_window
    
    
    return windowed_series