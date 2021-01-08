# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 12:11:00 2021

@author: ZR
"""
import My_Wheels.OS_Tools_Kit as OS_Tools
import numpy as np
import cv2

def AI_Calculator(graph_folder,
                  start_frame = 0,
                  end_frame = -1,
                  masks = 'No_Mask'
                ):
    '''
    This function is used to calculate average intensity variation. Masks can be given to calculate cells

    Parameters
    ----------
    graph_folder : (str)
        All graphs folder.
    start_frame : (int,optional)
        Start frame num. The default is 0.
    end_frame : (int,optional)
        End frame. The default is -1.
    masks : (2D_Array,optional)
        2D arrays. Input will be binary, so be careful. The default is None.

    Returns
    -------
    intensity_series : (Array)
        Return average intensity.

    '''
    #initialize
    all_tif_name = np.array(OS_Tools.Get_File_Name(graph_folder))
    used_tif_name = all_tif_name[start_frame:end_frame]
    frame_Num = len(used_tif_name)
    intensity_series = np.zeros(frame_Num,dtype='f8')
    graph_shape = np.shape(cv2.imread(used_tif_name[0],-1))
    #calculate mask
    if type(masks) == str:
        masks = np.ones(graph_shape,dtype = 'bool')
    elif masks.dtype != 'bool':
        masks = masks>(masks//2)
    pix_num = masks.sum()
    #calculate ai trains
    for i in range(frame_Num):
        current_graph = cv2.imread(used_tif_name[i],-1)
        masked_graph = current_graph*masks
        current_ai = masked_graph.sum()/pix_num
        intensity_series[i] = current_ai
    return intensity_series