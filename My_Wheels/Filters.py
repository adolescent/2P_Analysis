# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 16:14:23 2020

@author: ZR
A file of all filters. 
"""
from scipy.ndimage import correlate
import My_Wheels.Calculation_Functions as Calculator

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
        filter_design = 'butter',
        filter_para = (0.1,0.9)
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
    straight_power = data_train.mean()
    HP_prop = filter_para[0]
    LP_prop = filter_para[1]
    if filter_design == 'butter':
        if HP_prop != False and LP_prop != False:# Meaning we need band pass filter here.
            b, a = signal.butter(2, [HP_prop,LP_prop], 'bandpass')
            filtedData = signal.filtfilt(b, a, data_train)
        elif HP_prop == False and LP_prop == False:
            print('No filt.')
            filtedData = data_train
        elif LP_prop == False:
            b, a = signal.butter(2, HP_prop, 'highpass')
            filtedData = signal.filtfilt(b, a, data_train)
        elif HP_prop == False:
            b, a = signal.butter(2, LP_prop, 'lowpass')
            filtedData = signal.filtfilt(b, a, data_train)
            
        if HP_prop != False:
            filtedData = filtedData+straight_power
        
    elif filter_design == 'Fourier':
        print('FFT method developing..')
        filtedData = None
    else:
        raise IOError('filter design not finished yet.')
        
    return filtedData
    
