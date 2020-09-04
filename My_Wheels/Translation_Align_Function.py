# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 14:05:52 2020

@author: ZR
"""

import My_Wheels.OS_Tools_Kit as OS_Tools
import My_Wheels.List_Operation_Kit as List_Op
import My_Wheels.Graph_Operation_Kit as Graph_Tools
import numpy as np
import cv2
from My_Wheels.Alignment import Alignment
import time

#%%
def Translation_Alignment(
        all_folders,
        base_mode = 'global',
        input_base = np.array([[0,0],[0,0]]),
        align_range = 20,
        align_boulder = 20,
        before_average = True,
        average_std = 5,
        big_memory_mode = False,
        save_aligned_data = False
        ):
    '''
    
    This function will align all tif graphs in input folders. Only translation transaction here. Affine transformation need further discussion.
    
    Parameters
    ----------
    all_folders:(list)
        List of all tif folders, elements are strs.
    
    base_mode:('global',int,'input',optional. The default is 'global')
        How to select base frame. 'global': use global average as base. int: use average of specific run as base. 'input':Manually input base graph, need to be a 2D-Ndarray.
        
    input_base:(2D-Ndarray,optional. The default is none.)
        If base_mode = 'input', input_base must be given. This will be the base for alignment.
        
    align_range:(int,optional. The default is 20)
        Max pixel of alignment. 
        
    align_boulder:(int,optional. The default is 20)
        boulder cut for align. For different graph size, this variable shall be change.
        
    before_average:(bool,optional. The default is True)
        Whether before average is done. It can be set False to save time, on this case base graph shall be given.
        
    average_std:(float,optional. The default is 5)
        How much std you want for average graph generation. Different std can effect graph effect.
    
    big_memory_mode:(bool,optional. The default is False)
        If memory is big enough, use this mode is faster.
        
    save_aligned_data:(bool,optional. The default is False)
        Can be true only in big memory mode. This will save all aligned graph in a single 4D-Ndarray file.Save folder is the first folder.
    
        
    Returns
    -------
    bool
        Whether new folder is generated.
    
    '''
    start_time = time.time()
    # Step1, generate folders and file cycle.
    all_save_folders = List_Op.List_Annex(all_folders,['Results'])
    Aligned_frame_folders = List_Op.List_Annex(all_save_folders,['Aligned_Frames'])
    for i in range(len(all_save_folders)):
        OS_Tools.mkdir(all_save_folders[i])
        OS_Tools.mkdir(Aligned_frame_folders[i])
    Before_Align_Tif_Name = []
    for i in range(len(all_folders)):
        current_run_tif = OS_Tools.Get_File_Name(all_folders[i])
        Before_Align_Tif_Name.append(current_run_tif)
        
    # Step2, Generate average map before align.
    if before_average == True:
        print('Before run averaging ...')
        Before_Align_Dics = {}# This is the dictionary of all run averages. Keys are run id.
        total_graph_num = 0 # Counter of graph numbers.
        for i in range(len(Before_Align_Tif_Name)):
            current_run_graph_num = len(Before_Align_Tif_Name[i])
            total_graph_num += current_run_graph_num
            current_run_average = Graph_Tools.Average_From_File(Before_Align_Tif_Name[i])
            current_run_average = Graph_Tools.Clip_And_Normalize(current_run_average,clip_std = average_std)
            Before_Align_Dics[i] = (current_run_average,current_run_graph_num)# Attention here, data recorded as turple.
            Graph_Tools.Show_Graph(current_run_average, 'Run_Average',all_save_folders[i])# Show and save Run Average.
        # Then Use Weighted average method to generate global tif.
        global_average_graph = np.zeros(shape = np.shape(Before_Align_Dics[0][0]),dtype = 'f8')# Base on shape of graph
        for i in range(len(Before_Align_Tif_Name)):
            global_average_graph += Before_Align_Dics[i][0].astype('f8')*Before_Align_Dics[i][1]/total_graph_num
        global_average_graph = Graph_Tools.Clip_And_Normalize(global_average_graph,clip_std = average_std)
        # Then save global average in each run folder.
        for i in range(len(Before_Align_Tif_Name)):
            Graph_Tools.Show_Graph(global_average_graph, 'Global_Average', all_save_folders[i],show_time = 0)
        
    else:
        print('Before average Skipped.')
        
    # Step3, Core Align Function.
    print('Aligning...')
    if base_mode == 'global':
        base = global_average_graph
    elif base_mode == 'input':
        base = input_base
    elif type(base_mode) == int:
        base = Before_Align_Dics[base_mode][0]
    else:
        raise IOError('Invalid base mode.')
    # In big memory mode, save aligned_data in a dictionary file.
    if big_memory_mode == True:
        All_Aligned_Frame = {}
        for i in range(len(Before_Align_Tif_Name)):
            All_Aligned_Frame[i] = np.zeros(shape = )
        
    for i in range(len(Before_Align_Tif_Name)): # Cycle all runs
        for j in range(len(Before_Align_Tif_Name[i])): # Cycle all graphs in current run
            
    
    
    return True
