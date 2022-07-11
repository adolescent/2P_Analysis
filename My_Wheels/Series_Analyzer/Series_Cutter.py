# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 12:20:19 2022

@author: ZR

This function is used to cut series into small time windows.
"""
import numpy as np


def Series_Window_Slide(input_series,win_size = 300,win_step = 60):
    '''
    Generate a series of slide window data.

    Parameters
    ----------
    input_series : (ND array)
        Input series need to be cut.
    win_size : (int), optional
        Size of each window.(Frame) The default is 300.
    win_step : (int), optional
        Step of each window.(Frame) The default is 60.


    Returns
    -------
    windowed_series : (ND Array)
        Series of windowed graph.
    '''
    cell_num,frame_num = input_series.shape
    win_num = (frame_num-win_size)//win_step
    win_slide_frame = np.zeros(shape = (cell_num,win_size,win_num),dtype = 'f8')
    for i in range(win_num):
        c_window = input_series.iloc[:,i*win_step:i*win_step+win_size]
        win_slide_frame[:,:,i] = c_window

    return win_slide_frame