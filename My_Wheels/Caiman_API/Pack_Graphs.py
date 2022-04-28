# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 14:01:49 2022

@author: ZR
"""
from Decorators import Timer
import OS_Tools_Kit as ot
import tifffile as tif
import numpy as np
import cv2

@Timer
def Graph_Packer(data_folder_lists,save_folder,graph_shape = (512,512)):
    '''
    Pack all tif files in data folder into a tif stack.

    Parameters
    ----------
    data_folder : (list)
        List of folder of original data.
    save_folder : (str)
        Folder to save graphs into.
        
    Return
    ----------
    frame_num_lists : (list)
        List of frame numbers given. This is used to cut series back.
    
    '''
    
    frame_num_lists = []
    for i,c_folder in enumerate(data_folder_lists):
        c_tif_name = ot.Get_File_Name(c_folder)
        frame_num_lists.append(len(c_tif_name))
        c_tif_struct = np.zeros(shape = (len(c_tif_name),graph_shape[0],graph_shape[1]),dtype='u2')
        c_folder_name = c_folder.split('\\')[-1]
        for i in range(len(c_tif_name)):
            c_graph = cv2.imread(c_tif_name[i],-1)
            c_tif_struct[i,:,:] = c_graph
        tif.imwrite(save_folder+r'\\'+c_folder_name+'.tif',c_tif_struct)
        
        
    return frame_num_lists