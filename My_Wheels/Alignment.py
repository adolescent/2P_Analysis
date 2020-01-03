# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 15:41:18 2019

@author: ZR

Align Function. This function is the CORE of Data Pre processing, so list this alone.
"""

import numpy as np
import My_Wheels.Graph_Operation_Kit as Graph_Tools


#%% Function is designed for Alignment only, API not easy to use directly.
def Bais_Correlation(extended_base,extended_target,align_range):
    """
    Calculate Bais use frequency correlation method,return x/y bais
    
    Parameters
    ----------
    extended_base : (2D Array)
        base graph, extended with zero .
    extended_target : (2D Array)
        target graph, extended with zero pad.
    align_range : (int)
        maxiunm pix of align.

    Returns
    -------
    x_bais : (int)
        Best match x bais, positive x means target shall move right.
    y_bais : (int)
        Best match y bais, positive y means target shall move down.

    """
    target_fft = np.fft.fft(extended_target)
    base_fft = np.fft.fft(extended_base)
    conv2 = np.real(np.fft.ifft2(target_fft*base_fft)) # Convolution between base & target. This calculation method will be faster than direct conv function. 
    conv_height,conv_width = np.shape(conv2)
    y_center,x_center = (int((conv_height-1)/2),int((conv_width-1)/2)) # Center of concolution matrix, if perfect match, this center will have biggest value.
    find_location = conv2[(y_center-align_range):(y_center+align_range),(x_center-align_range):(x_center+align_range)] # range of peak find, this matrix will determine x/y bais.
    y_bais = np.where(find_location ==np.max(find_location))[0][0] -align_range # find y bais  
    x_bais = np.where(find_location ==np.max(find_location))[1][0] -align_range # find x bais 
    return x_bais,y_bais
    
#%% Main Align Function
def Alignment(base_graph,target_graph,boulder = 20,align_range = 20):
    """
    Move target graph to match base graph. fill blanks with line median.

    Parameters
    ----------
    base_graph : (2D Array)
        Base Graph. All target will be align to this one. Use global average usually.
    target_graph : (2D Array)
        Current Graph. This graph will be moved.
    boulder : (int), optional
        Use center to align graphs, base will cut a boulder. The default is 20.
    align_range : (int), optional
        Maximun pixel of Align. The default is 20.
        
    Returns
    -------
    x_bais : (int)
        X bais. Positive x_bais means target graph shall move right to match base.
    y_bais : (int)
        Y bais. Positive y_bais means target graph shall move down to match base.
    aligned_graph : (ndarray)
        moved graph. 

    """
    
    target_boulder = int(boulder+np.floor(align_range*1.5))# target will cut with a bigger boulder.
    cutted_target = Graph_Tools.Graph_Cut(target_graph, [target_boulder,target_boulder,target_boulder,target_boulder])
    target_height,target_width = np.shape(cutted_target)
    cutted_base = Graph_Tools.Graph_Cut(base_graph,[boulder,boulder,boulder,boulder])
    base_height,base_width = np.shape(cutted_base)
    extended_target = np.pad(np.rot90(cutted_target,2),((0,base_height-1),(0,base_width-1)),'constant') # Extend graph here to make sure Returned FFT Matrix have same shape, making comparation easier.
    extended_base = np.pad(cutted_base,((0,target_height-1),(0,target_width-1)),'constant')
    x_bais,y_bais = Bais_Correlation(extended_base, extended_target, align_range)
    temp_aligned_graph = np.pad(target_graph,((align_range+y_bais,align_range-y_bais),(align_range+x_bais,align_range-x_bais)),'median') # Fill target graph with median graphs
    aligned_graph = temp_aligned_graph[align_range:-align_range,align_range:-align_range] # Cut Boulder, return moved graph.
    
    return x_bais,y_bais,aligned_graph
