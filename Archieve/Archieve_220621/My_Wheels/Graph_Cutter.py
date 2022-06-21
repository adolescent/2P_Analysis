# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 15:06:59 2020

@author: ZR
"""
import numpy as np
import cv2
import My_Wheels.Graph_Operation_Kit as Graph_Tools

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
    length_after_cut = length-boulder*2
    width_after_cut = width-boulder*2
    # Get shape of cutted graph
    cutted_length = length_after_cut//cut_shape[0]
    cutted_width = width_after_cut//cut_shape[1]
    after_size = (cutted_length,cutted_width)
    graph_location_list = []
    # cycle all fractions.
    cutted_graph_dics = {}
    current_fraction_id = 0
    for i in range(cut_shape[1]):
        for j in range(cut_shape[0]):
            left_up_point = (boulder+j*cutted_length,boulder+i*cutted_width)
            graph_location_list.append(left_up_point)
            current_image = input_graph[left_up_point[0]:left_up_point[0]+cutted_length,left_up_point[1]:left_up_point[1]+cutted_width]
            cutted_graph_dics[current_fraction_id] = current_image
            current_fraction_id +=1
            
    # Then draw schametic graph, and show id on it. cv2 location id sequence is xy!!
    schametic = Graph_Tools.Clip_And_Normalize(input_graph,clip_std = 10,bit = 'u1')
    for i in range(cut_shape[0]+1): # Draw horizontal lines
        cv2.line(schametic,(boulder,boulder+i*cutted_length),(width-boulder,boulder+i*cutted_length),(255),2)
    for i in range(cut_shape[1]+1): #...And vertical lines
        cv2.line(schametic,(boulder+i*cutted_width,boulder),(boulder+i*cutted_width,length-boulder),(255),2)
    for i in range(len(graph_location_list)): # And put graph id on them.
        text_loc = (graph_location_list[i][1]+cutted_width//2,graph_location_list[i][0]+cutted_length//2)
        cv2.putText(schametic,str(i),text_loc,cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255),1)
        
    return schametic,graph_location_list,after_size,cutted_graph_dics