# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 16:25:06 2020

@author: ZR
Alignment based on affine transformation.
This is used to align graph with bigger tremble, theoretically can fix stretch and rotation of images

"""

import cv2
import numpy as np
import My_Wheels.OS_Tools_Kit as OS_Tools
import My_Wheels.Graph_Operation_Kit as Graph_Tools
import My_Wheels.Filters as Filters
import warnings
#%% Old Core function of affine.
# Function Out of Date.
# This function is really OLD!! Will be out of date very sooon.
# =============================================================================
# def Affine_Core(
#         base,
#         target,
#         gain = 20,
#         max_point = 100000,
#         good_match = 0.15,
#         dist_lim = 40,
#         Filter = True,
#         match_checker = 1,
#         ):
#     '''
#     Use ORB method to do affine correlation.
#     Remenber, this method will change all graph into u1 to calculate, using normalization method.
#     But output method will not be changed.
#     
#     Parameters
#     ----------
#     base : (2D Array,dtype = 'u1')
#         Base graph. Usually input as uint8 type.
#     target : (2D Array,dtyoe = 'u2')
#         Target graph. This graph will be deformed. Also is uint 16 dtype.
#     gain : (int,optional)
#         Gain of target graph. This will be very useful when converted into u1. 
#     max_point : (int), optional
#         Max number of feature selection. The default is 100000.
#     good_match : (float), optional
#         Good match percentage. Percentage of point be choosen to calculate h matrix. The default is 0.15.
#     dist_lim : (int,optional)
#         Limitation of distance of match. match over this will be regarded as unmatch.
#         
#     Returns
#     -------
#     matched_graph : (2D Array)
#         Deformed graph to match target.
#     h : (3*3Matrix)
#         Affine transformation matrix.
# 
#     '''
#     # Check data type.
#     if base.dtype != np.dtype('u1'):
#         raise IOError('Base graph dtype shall be u1!')
#     if target.dtype != np.dtype('u2'):
#         raise IOError('Target graph dtype shall be u2!')
#     # Change graph data type. Only 8bit 1channel graph is allowed.
#     if Filter == True:
#         target_filted = Filters.Filter_2D(target,HP_Para=False)
#     else:
#         target_filted = target
#     target_8bit = np.clip((target_filted.astype('f8')*gain/256),0,255).astype('u1')
#     # Detect ORB features and compute descriptors.
#     orb = cv2.ORB_create(max_point)
#     keypoints1, descriptors1 = orb.detectAndCompute(target_8bit, None)
#     keypoints2, descriptors2 = orb.detectAndCompute(base, None)
#     # Match features.
#     matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
#     matches = matcher.match(descriptors1, descriptors2, None)
#     # Sort matches by score
#     matches.sort(key=lambda x: x.distance, reverse=False)
#     # Remove not so good matches
#     numGoodMatches = int(len(matches)*good_match)
#     matches = matches[:numGoodMatches]
#     while (matches[-1].distance>dist_lim):
#         matches.pop(-1)
#     # Extract location of good matches
#     points1 = np.zeros((len(matches), 2), dtype=np.float32)
#     points2 = np.zeros((len(matches), 2), dtype=np.float32)
#     for i, match in enumerate(matches):
#         points1[i, :] = keypoints1[match.queryIdx].pt
#         points2[i, :] = keypoints2[match.trainIdx].pt
#     # Find homography
#     h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
#     # h Check here to avoid bad mistake. This part can be revised and discussed.
#     if abs(h[0,1])>match_checker:
#         raise IOError('Bad match, please check parameters.')
#     
#     height,width = base.shape
#     matched_graph = cv2.warpPerspective(target, h, (width, height))
#     
#     return matched_graph,h
# =============================================================================

#%% Affine match with point restriction.
def Affine_Core_Point_Equal(
        target,
        base,
        targ_gain = 20,
        max_point = 50000,
        good_match_prop = 0.3,
        sector_num = 4,
        dist_lim = 200,
        Filter = True,
        match_checker = 1
        ):
    '''
    Core function of affine align, will selece equal spetial point vertically.

    Parameters
    ----------
    target : (2D Array, dtype = u1/u2)
        The graph will be aligned.
    base : (2D Array, dtype = u1/u2)
        Base graph. Target will be aligned to this.
    targ_gain : (int), optional
        Gain used to do align. The default is 20.
    max_point : (int), optional
        Max number of feature points. The default is 50000.
    good_match_prop : (float), optional
        Propotion of good match in all matches. The default is 0.3.
    sector_num : (int), optional
        Cut graph vertically in several sections, all section have equal points. The default is 4.
    dist_lim : (int), optional
        Distance limitation of 2 matches, match above this will be ignored. The default is 200.
    Filter : (bool), optional
        Whether we do space filter. The default is True.
    match_checker : (float), optional
        A checker for h matrix. Bigger checher means we tolerate more graph deformation. The default is 1.


    Returns
    -------
    matched_graph : (2D Array)
        Deformed graph. Shape will be the same as base graph.
    h : TYPE
        DESCRIPTION.

    '''
    height,width = base.shape
    # Check data type.
    if base.dtype == np.dtype('u2'):
        base = (base/256).astype('u1')
    elif base.dtype != np.dtype('u1'):
        raise IOError('Base graph dtype shall be u1 or u2.')
    if target.dtype == np.dtype('u1'):
        target = target.astype('u2')*256
    elif target.dtype == np.dtype('u2'):
        target = target
    else:
        raise IOError('Target graph dtype shall be u1 or u2!')
    # Change graph data type. Only 8bit 1channel graph is allowed.
    if Filter == True:
        target_filted = Filters.Filter_2D(target,HP_Para=False)
    else:
        target_filted = target
    target_8bit = np.clip((target_filted.astype('f8')*targ_gain/256),0,255).astype('u1')
    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(max_point)
    keypoints1, descriptors1 = orb.detectAndCompute(target_8bit, None)
    keypoints2, descriptors2 = orb.detectAndCompute(base, None)
    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
    # Then eliminate match with bigger dist.
    matches.sort(key=lambda x: x.distance, reverse=False)
    while (matches[-1].distance>dist_lim):
        matches.pop(-1)
    # Then get num of good matches and distribute them into quaters.
    good_match_num = round(len(matches)*good_match_prop)
    max_point_per_sector = good_match_num//sector_num
    sector_height = height//sector_num
    sector_counter = np.zeros(sector_num)
    used_matches = []
    for i in range(len(matches)):
        current_y = keypoints2[matches[i].trainIdx].pt[1] # Use y loc in base graph as indicator.
        current_sector = int(current_y//sector_height)
        if sector_counter[current_sector]<max_point_per_sector:
            used_matches.append(matches[i])
            sector_counter[current_sector] +=1
    # Extract location of good matches
    points1 = np.zeros((len(used_matches), 2), dtype=np.float32)
    points2 = np.zeros((len(used_matches), 2), dtype=np.float32)
    for i, match in enumerate(used_matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)  
    # h Check here to avoid bad mistake. This part can be revised and discussed.
    if abs(h[0,1])>match_checker:
        warnings.warn('Bad match, please check parameters.',UserWarning)
    height,width = base.shape
    matched_graph = cv2.warpPerspective(target, h, (width, height))
    #matched_graph = np.maximum(matched_graph,1)
    return matched_graph,h
        


# =============================================================================
# Function Out of Dated.
# #%% Practical affine function circulation.
# def Affine_Graph_Aligner(
#         data_folder,
#         base_graph,
#         max_point = 50000,
#         good_match_prop = 0.3,
#         dist_lim = 120,
#         match_checker = 1
#         ):
#     OS_Tools.mkdir(data_folder+r'\Results')
#     aligned_graph_folder = data_folder+r'\Results\Affine_Aligned_Graphs'
#     OS_Tools.mkdir(aligned_graph_folder)
#     all_tif_name = OS_Tools.Get_File_Name(data_folder)
#     h_dic = {}
#     for i in range(len(all_tif_name)):
#         current_target = cv2.imread(all_tif_name[i],-1)
#         aligned_graph,h = Affine_Core_Point_Equal(current_target,base_graph,max_point = max_point,good_match_prop = good_match_prop,dist_lim =dist_lim,Filter = True,match_checker = match_checker)
#         Graph_Tools.Show_Graph(aligned_graph,all_tif_name[i].split('\\')[-1], aligned_graph_folder,show_time = 0,graph_formation= '')
#         h_dic[i] = h
#     aligned_all_graph_name = OS_Tools.Get_File_Name(aligned_graph_folder)
#     average_after = Graph_Tools.Average_From_File(aligned_all_graph_name)
#     average_after = Graph_Tools.Clip_And_Normalize(average_after,clip_std = 5)
#     Graph_Tools.Show_Graph(average_after, 'Average_After_Affine', data_folder+r'\Results')
#     OS_Tools.Save_Variable(data_folder+r'\Results', 'Align_Paras', h_dic)
#     return h_dic
# =============================================================================
#%% Last, use gaussian average to do an align. BIG MEMORY required.
def Affine_Aligner_Gaussian(
        data_folder,
        base_graph,
        window_size = 1,          
        max_point = 50000,
        good_match_prop = 0.3,
        dist_lim = 120,
        match_checker = 1,
        sector_num = 4,
        write_file = False
        ):
    save_folder = data_folder+r'\Results'
    aligned_tif_folder = save_folder+r'\Affined_Frames'
    OS_Tools.mkdir(save_folder)
    OS_Tools.mkdir(aligned_tif_folder)
    
    all_tif_name = OS_Tools.Get_File_Name(data_folder)
    graph_num = len(all_tif_name)
    graph_shape = cv2.imread(all_tif_name[0],-1).shape
    height,width = graph_shape
    origin_tif_matrix = np.zeros(shape = graph_shape+(graph_num,),dtype = 'u2')
    # Read in all tif name.
    for i in range(graph_num):
        origin_tif_matrix[:,:,i] = cv2.imread(all_tif_name[i],-1)
    # Then get window slipped average graph.
    if window_size == 1:
        slipped_average_matrix = origin_tif_matrix
    else:
        slipped_average_matrix = Filters.Window_Average(origin_tif_matrix,window_size = window_size)
    # Use slip average to get deformation parameters.
    aligned_tif_matrix = np.zeros(shape = origin_tif_matrix.shape,dtype = 'u2')
    h_dic = {} # Deformation parameters
    for i in range(graph_num):
        target = slipped_average_matrix[:,:,i]
        _,current_h = Affine_Core_Point_Equal(target, base_graph,max_point=max_point,good_match_prop = good_match_prop,sector_num = sector_num,dist_lim = dist_lim,match_checker = match_checker)
        h_dic[i] = current_h
        current_deformed_graph = cv2.warpPerspective(origin_tif_matrix[:,:,i], current_h, (width, height))
        Graph_Tools.Show_Graph(current_deformed_graph,all_tif_name[i].split('\\')[-1], aligned_tif_folder,show_time = 0,graph_formation= '')
        aligned_tif_matrix[:,:,i] = current_deformed_graph
    OS_Tools.Save_Variable(save_folder, 'Deform_H', h_dic)
    if write_file == True:
        OS_Tools.Save_Variable(save_folder, 'Affine_Aligned_Graphs', aligned_tif_matrix)
    # At last, generate average graphs
    graph_before_align = origin_tif_matrix.mean(axis = 2).astype('u2')
    graph_after_align = aligned_tif_matrix.mean(axis = 2).astype('u2')
    graph_before_align = Graph_Tools.Clip_And_Normalize(graph_before_align,clip_std = 5)
    graph_after_align = Graph_Tools.Clip_And_Normalize(graph_after_align,clip_std = 5)
    Graph_Tools.Show_Graph(graph_before_align, 'Graph_Before_Affine', save_folder)
    Graph_Tools.Show_Graph(graph_after_align, 'Graph_After_Affine', save_folder)
    return True


