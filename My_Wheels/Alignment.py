# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 15:41:18 2019

@author: ZR

Align Function. This function is the CORE of Data Pre processing, so list this alone.
"""

import numpy as np
import My_Wheels.Graph_Operation_Kit as Graph_Tools


#%% Method Function 

#%% Main Align Function
def Alignment(base_graph,target_graph,boulder = 20,align_range = 20,dtype = 'u2'):
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
    dtype = ('u1','u2','f8'),optional
        Data Type of graphs. Attention here, input & output shall be the same.

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
    
    

    return x_bais,y_bais,aligned_graph