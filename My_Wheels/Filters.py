# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 16:14:23 2020

@author: ZR
A file of all filters. 
"""
from scipy.ndimage import correlate
import My_Wheels.Calculation_Functions as Calculator
import numpy as np

#%% 2D Filters
def Filter_2D_Kenrel(graph,kernel):
    '''
    Kenrel function of all filters. We correlate graph with kernel to do the job.

    Parameters
    ----------
    graph : (2D Array)
        Input graph.
    kernel : (2D Array)
        Kernel function of filter.

    Returns
    -------
    filtered_graph : (2D Array, dtype = 'f8')
        Filtered graph.

    '''
    graph = graph.astype('f8')
    kernel = kernel.astype('f8')
    filtered_graph = correlate(graph,kernel,mode = 'reflect')
    return filtered_graph


def Filter_2D(
        graph,
        LP_Para = ([5,5],1.5),
        HP_Para = ([30,30],10),
        filter_method = 'Gaussian'
        ):
    '''
    Filt input graph. Both HP and LP included, but both can be cancled by set to 'False'.
    This filter will reserve straight power, meaning we don't change global average.

    Parameters
    ----------
    graph : (2D Array)Y
        Input graph. Any data type is allowed, and will return same dtype.
    LP_Para : (turple or False), optional
        False will cancel this calculation. Lowhpass parameter. The default is ((5,5),1.5).
    HP_Para : (turple or False), optional
        False will cancel this calculation. Highpass parameter. The default is ((30,30),10).
    filter_method : (str), optional
        Method of filter, can be updated anytime. The default is 'Gaussian'.
        'Gaussian': Gaussian filter. Attention:Gaussian method can be very slow when it came to big HP Para!!
        'Fourier': Use FFT method to get filtered graph.

    Returns
    -------
    filtered_graph : (2D Array)
        Filtered graph. Dtype as input.

    '''
    origin_dtype = graph.dtype
    graph = graph.astype('f8')
    if filter_method == 'Gaussian': # Do Gaussian Filter.
        if LP_Para != False:
            LP_kernel = Calculator.Normalized_2D_Gaussian_Generator(LP_Para)
            LP_filted_graph = Filter_2D_Kenrel(graph, LP_kernel)
        else:
            LP_filted_graph = graph
        
        if HP_Para != False:
            HP_kernel = Calculator.Normalized_2D_Gaussian_Generator(HP_Para)
            Low_Band_graph = Filter_2D_Kenrel(graph, HP_kernel)
            BP_filted_graph = LP_filted_graph - Low_Band_graph
            straight_power = Low_Band_graph.mean()
            BP_filted_graph = BP_filted_graph+straight_power
        else:
            BP_filted_graph = LP_filted_graph
        
        filtered_graph = BP_filted_graph.astype(origin_dtype)# Add straight power on.
        
    elif filter_method == 'Fourier':
        print('Function Developing...')
        filtered_graph = None
    else:
        raise IOError('Filter method not supported...Yet.')
    
    return filtered_graph

#%% Signal Filters
from scipy import signal
def Signal_Filter(
        data_train,
        order = 5,
        filter_design = 'butter',
        filter_para = (0.1,0.9),method = 'pad',padtype='odd',dc_keep = True
        ):
    '''
    Filt Signal and return filted train.
    Straight Power reserved.
    Parameters
    ----------
    data_train : (Np Array)
        Input data train. Need to be an float64 array.
    filter_design : (str), optional
        Method of filter design. The default is 'butter'.
    filter_para:(turple),optional
        Each element can be set False to skip HP or LP. This input give the selected freq propotion. For 20Hz capture, (0.1,0.9) 1~9Hz.

    Returns
    -------
    filtedData : TYPE
        DESCRIPTION.

    '''
    straight_power = float(data_train.mean())
    HP_prop = filter_para[0]
    LP_prop = filter_para[1]
    # win = signal.hamming(len(data_train))
    data_train_win = data_train
    if filter_design == 'butter':
        if HP_prop != False and LP_prop != False:# Meaning we need band pass filter here.
            b, a = signal.butter(order, [HP_prop,LP_prop], 'bandpass')
            filtedData = signal.filtfilt(b, a, data_train_win,method = method,padtype = padtype)
        elif HP_prop == False and LP_prop == False:
            #print('No filt.')
            filtedData = data_train_win
        elif LP_prop == False:
            b, a = signal.butter(order, HP_prop, 'highpass')
            filtedData = signal.filtfilt(b, a, data_train_win,method = method,padtype = padtype)
        elif HP_prop == False:
            b, a = signal.butter(order, LP_prop, 'lowpass')
            filtedData = signal.filtfilt(b, a, data_train_win,method = method,padtype = padtype)
            
        if HP_prop != False:
            if dc_keep == True:
                filtedData = filtedData+straight_power
        
    elif filter_design == 'Fourier':
        print('FFT method developing..')
        filtedData = None
    else:
        raise IOError('filter design not finished yet.')
        
    return filtedData


def Signal_Filter_v2(series,HP_freq,LP_freq,fps,keep_DC = True,order = 5):
    DC_power = float(series.mean())
    nyquist = 0.5 * fps
    low = LP_freq / nyquist
    high = HP_freq / nyquist
    filtedData = series
    # do low pass first.
    if LP_freq != False:
        b, a = signal.butter(order, low, 'lowpass')
        filtedData = signal.filtfilt(b, a,filtedData,method = 'pad',padtype ='odd')
    if HP_freq != False:
        b, a = signal.butter(order, high, 'highpass')
        filtedData = signal.filtfilt(b, a,filtedData,method = 'pad',padtype ='odd')

    # b, a = signal.butter(order, [low, high], btype='bandpass')
    # filtered_data = signal.filtfilt(b, a, series,method = 'pad',padtype='odd')
    if keep_DC == True:
        filtedData += DC_power

    return filtedData

#%% Windows slip
def Window_Average(
        data_matrix,
        window_size = 5,
        window_method = 'Gaussian'
        ):
    '''
    Average data matrix with given 

    Parameters
    ----------
    data_matrix : TYPE
        DESCRIPTION.
    window_size : TYPE, optional
        DESCRIPTION. The default is 5.
    window_method : TYPE, optional
        DESCRIPTION. The default is 'Gaussian'.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    origin_dtype = data_matrix.dtype
    if window_size%2 == 0:
        raise IOError('Window Size need to be odd!.')
    graph_num = data_matrix.shape[2]
    extended_graph_num = graph_num+window_size-1
    # Use reflect boulder, extend data matrix to fit for window.
    frame_extend = int((window_size-1)/2)
    extended_graph_matrix = np.zeros(shape = (data_matrix.shape[0],data_matrix.shape[1],extended_graph_num),dtype = origin_dtype)
    extended_graph_matrix[:,:,frame_extend:extended_graph_num-frame_extend] = data_matrix
    for i in range(frame_extend):
        extended_graph_matrix[:,:,frame_extend-i-1]=extended_graph_matrix[:,:,frame_extend+i+1]# head reflection
        extended_graph_matrix[:,:,extended_graph_num-frame_extend+i]=extended_graph_matrix[:,:,extended_graph_num-frame_extend-i-2]# Tail reflection
    # Get window kernel function, use this 
    slip_window = np.zeros((window_size),dtype = 'f8')
    if window_method == 'Average':
        slip_window[:] = 1/window_size
    elif window_method == 'Gaussian':
        slip_window = Calculator.Normalized_1D_Gaussian_Generator(window_size,window_size/5)
    else:
        raise IOError('Window method not supported.')
    # Then get the slip average.
    slipped_data_matrix = np.zeros(data_matrix.shape,dtype = origin_dtype) # remain dtype unchanged. 
    reshapped_data = extended_graph_matrix.reshape(-1,extended_graph_num)
    for i in range(graph_num):
        current_slice = reshapped_data[:,i:i+window_size]
        current_frame = np.average(current_slice,axis = 1,weights=slip_window).reshape(data_matrix.shape[0],data_matrix.shape[1])
        slipped_data_matrix[:,:,i] = current_frame
    averaged_series = slipped_data_matrix
    return averaged_series
