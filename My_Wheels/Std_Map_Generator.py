# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 13:30:59 2020

@author: zhang

Generate std map for input graph sets. This process is important for spontaneous cell find.
"""
import numpy as np
import cv2
def Std_Map_Generator(all_tif_name,clip_std = 2.5,min_brightness = 0,return_type = 'u1'):
    """
    Generate std map of input tifs, useful for cell finding.

    Parameters
    ----------
    all_tif_name : (list)
        List of used all tif names.
    clip_std : (float), optional
        How many std used to clip graph. The default is 2.5.
    min_brightness: (int),optinal
        If mean < minbrightness, return 0 on final graph. The default is 0.
    return_type : ('origin','normalize','u1'), optional
        Data type of returned graph. The default is 'u1'.

    Returns
    -------
    std_map : (2D Array)
        Std distribution map. brighter means higher std.

    """
    Frame_Num = len(all_tif_name)
    height,width = np.shape(cv2.imread(all_tif_name[0],-1))
    all_graph_matrix = np.zeros(shape = (height,width,Frame_Num),dtype = 'u2')
    # Step 1, read in all graph
    for i in range(Frame_Num):
        current_graph = cv2.imread(all_tif_name[i],-1)
        all_graph_matrix[:,:,i] = current_graph
    # Step 2, get origin std map
    reshaped_all_graph_matrix = all_graph_matrix.reshape(-1,Frame_Num)
    del all_graph_matrix
    origin_std_map = np.std(reshaped_all_graph_matrix,axis = 1).reshape(height,width)
    # Step 3, get brightness mask, used for final process
    mean_map = np.mean(reshaped_all_graph_matrix,axis = 1).reshape(height,width)
    bright_mask = mean_map>min_brightness
    del reshaped_all_graph_matrix
    # Step 4, clip and mask output graph.    
    map_std = np.std(origin_std_map)
    map_mean = np.mean(origin_std_map)
    clipped_std_map = np.clip(origin_std_map,map_mean-clip_std*map_std,map_mean+clip_std*map_std)
    std_map = clipped_std_map*bright_mask.astype('f8')
    # Last, change output type according to return_type.
    if return_type == 'origin':
        pass
    elif return_type == 'normalize':
        std_map = (std_map-std_map.min())/(std_map.max()-std_map.min())
    elif return_type == 'u1':
        std_map = ((std_map-std_map.min())/(std_map.max()-std_map.min())*255).astype('u1')
    return std_map
