# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 15:55:28 2021

@author: ZR
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import OS_Tools_Kit as ot


def FFT_Power(input_series,signal_name = 'Input',normalize = False,fps = 1.301):
    '''
    Single FFT Power spectrum

    Parameters
    ----------
    input_series : (1D-Nd Array)
        Input signal. Only 1D Array accepted.
    fps : (Hz), optional
        Capture frequency of signal. The default is 1.301.
    signal_name : (str), optional
        Name of input signal. The default is 'Input'.
    normalize : (bool),optional
        Whether we return spectrum of normalized spectrum.

    Returns
    -------
    Power_Spectrum : (DataFrame)
        Power spectrum of input series.

    '''
    spec_size = round(len(input_series)/2)
    raw_fft = np.fft.fft(input_series)
    raw_power = abs(raw_fft)[:spec_size]
    if normalize == True:
        raw_power = raw_power/raw_power.sum()
    freq_list = np.linspace(0,fps/2,num = spec_size)
    Power_Spectrum = pd.DataFrame({signal_name:raw_power[:spec_size]})
    new_index = {}
    for i,c_freq in enumerate(freq_list):
        new_index[i] = round(c_freq,7)
    Power_Spectrum = Power_Spectrum.rename(index = new_index)
    return Power_Spectrum

def FFT_Window_Slide(whole_train,window_length = 300,window_step = 60,fps=1.301): 
    
    window_frame_length = int(window_length*fps)
    window_frame_step = int(window_step*fps)
    frame_num = len(whole_train)
    window_num = 1+int((frame_num-window_frame_length)/window_frame_step)
    slided_power_spectrum = FFT_Power(input_series = whole_train[0:window_frame_length],signal_name = 0,fps = fps)
    for i in range(0,window_num):
        c_window = (i*window_frame_step,i*window_frame_step+window_frame_length)
        c_series = whole_train[c_window[0]:c_window[1]]
        c_power = FFT_Power(c_series,signal_name=i,fps = fps)
        slided_power_spectrum = pd.concat([slided_power_spectrum,c_power],axis=1)
    
    return slided_power_spectrum