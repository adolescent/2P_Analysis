# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 16:14:23 2020

@author: ZR
A file of all filters. 
"""
from scipy.ndimage import correlate
import My_Wheels.Calculation_Functions as Calculator


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
        HP_Para = ([5,5],1.5),
        LP_Para = ([100,100],20),
        filter_method = 'Gaussian'
        ):
    '''
    Filt input graph. Both HP and LP included, but both can be cancled by set to 'False'

    Parameters
    ----------
    graph : (2D Array)
        Input graph. Any data type is allowed, and will return same dtype.
    HP_Para : (turple or False), optional
        False will cancel this calculation.Highpass parameter. The default is ((5,5),1.5).
    LP_Para : (turple or False), optional
        False will cancel this calculation.Lowpass parameter. The default is ((100,100),20).
    filter_method : (str), optional
        Method of filter, can be updated anytime. The default is 'Gaussian'.

    Returns
    -------
    filtered_graph : (2D Array)
        Filtered graph. Dtype same as input.

    '''
    origin_dtype = graph.dtype
    graph = graph.astype('f8')
    # First, do HP filter.
    if HP_Para != False:
        if filter_method == 'Gaussian':
            HP_kernel = Calculator.Normalized_2D_Gaussian_Generator(HP_Para)
            HP_filted_graph = Filter_2D_Kenrel(graph, HP_kernel)
        else:
            raise IOError('Filter method not supported....Yet')
    else:
        print('No HP allowed')
        HP_filted_graph = graph
    # Then, do LP filter.
    if LP_Para != False:
        pass
    else:
        print('No ')
    
    

    return filtered_graph