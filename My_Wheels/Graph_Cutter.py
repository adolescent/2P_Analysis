# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 15:06:59 2020

@author: ZR
"""
import numpy as np

def Graph_Cutter(
        input_graph,
        boulder = 20,
        cut_shape = (4,4),
        ):
    """
    Cut a whole graph into small pieces.

    Parameters
    ----------
    input_graph : (2D_NdArray)
        Input graph of calculation. 
    boulder : (int), optional
        Boulder cut to get rid of move error. The default is 20.
    cut_shape : (turple), optional
        Shape you want to cut graph into. The default is (4,4).


    Returns
    -------
    schametic : (2D-NdArray,dtype = 'u1')
        Schamatic graph of cut method, with ID on graph.
    graph_lacation_dics : (list)
        List of left upper coordinate of each graph.
    after_size : (turple)
        Size of small graph after cut. 2-element turple, yx.
    cutted_graph_dics : (Dic)
        Dictionary of cutted graphs.

    """
    length,width = np.shape(input_graph)
    cutted_graph = input_graph[boulder:lenth-boulder,boulder:width-boulder]
    
    return schametic,graph_location_list,after_size,cutted_graph_dics