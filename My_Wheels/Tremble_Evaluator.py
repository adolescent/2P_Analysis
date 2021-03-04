# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 10:54:45 2020

@author: ZR
This file used to Evaluate align quality. Through mass center 

"""
from My_Wheels.Graph_Cutter import Graph_Cutter
import My_Wheels.OS_Tools_Kit as OS_Tools
import My_Wheels.Graph_Operation_Kit as Graph_Tools
import numpy as np
import cv2
from skimage import filters
from skimage.measure import regionprops
import matplotlib.pyplot as plt


def Tremble_Evaluator(
        data_folder,
        ftype = '.tif',
        boulder_ignore = 20,
        cut_shape = (4,4),
        mask_thres = 'otsu'
        ):
    save_folder = data_folder+r'\Results'
    all_file_name = OS_Tools.Get_File_Name(data_folder,ftype)
    template = cv2.imread(all_file_name[0],-1)
    origin_dtype = template.dtype
    graph_shape = template.shape
    graph_num = len(all_file_name)
    origin_graph_matrix = np.zeros(shape = graph_shape+(graph_num,),dtype = origin_dtype)
    for i in range(graph_num):
        origin_graph_matrix[:,:,i] = cv2.imread(all_file_name[i],-1)
    average_graph = origin_graph_matrix.mean(axis = 2).astype('u2')
    # Show schematic of cutted graph.
    schematic,_,_,_ = Graph_Cutter(average_graph,boulder_ignore,cut_shape)
    Graph_Tools.Show_Graph(schematic, 'Cut Schematic', save_folder)
    # Then,save cutted graphs into dics.
    cutted_graph_dic = {}
    fracture_num = cut_shape[0]*cut_shape[1]
    for i in range(fracture_num):# initialize cut dics.
        cutted_graph_dic[i]=[]
    for i in range(graph_num):# Cycle all graphs
        current_graph = origin_graph_matrix[:,:,i]
        _,_,_,cutted_graphs = Graph_Cutter(current_graph,boulder_ignore,cut_shape)
        for j in range(fracture_num):# save each fracture
            cutted_graph_dic[j].append(cutted_graphs[j])
    # Calculate graph center of each fracture trains. Use weighted center.
    all_frac_center = np.zeros(shape = (fracture_num,graph_num,2),dtype = 'f8')
    for i in range(fracture_num):
        current_frac = cutted_graph_dic[i]
        for j in range(graph_num):
            current_graph = current_frac[j]
            if mask_thres == 'otsu':
                thres = filters.threshold_otsu(current_graph)
            elif (type(mask_thres) == int or type(mask_thres) == float):
                thres = mask_thres
            else:
                raise IOError('Invalid mask threshold.')
            mask = (current_graph > thres).astype(int)
            properties = regionprops(mask, current_graph)
            current_mc = properties[0].weighted_centroid
            all_frac_center[i,j,:] = current_mc #In sequence YX
    # Then, calculate final mass center location & graph ploting.
    fig, ax = plt.subplots(3,3,figsize = (20,20))
    ax[1,2].plot([1, 2, 3, 4], [1, 4, 9, 16])
    t2 = np.arange(0, 5, 0.02)
    plt.subplot(339)
    plt.plot(t2, np.cos(2 * np.pi * t2), 'r--')
    
    