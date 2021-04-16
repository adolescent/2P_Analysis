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
        save_aligned_data = False,
        graph_shape = (512,512),
        timer = True
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
        
    graph_shape:(2-element-turple,optional. The default is (512,512))
        Shape of graphs. All input graph must be in same shape.
        
    timer:(bool,optional. The default is True)
        Show runtime of function and each procedures.
    
        
    Returns
    -------
    bool
        Whether new folder is generated.
    
    '''
    time_tic_start = time.time()
    #%% Step1, generate folders and file cycle.
    all_save_folders = List_Op.List_Annex(all_folders,['Results'])
    Aligned_frame_folders = List_Op.List_Annex(all_save_folders,['Aligned_Frames'])
    for i in range(len(all_save_folders)):
        OS_Tools.mkdir(all_save_folders[i])
        OS_Tools.mkdir(Aligned_frame_folders[i])
    Before_Align_Tif_Name = []
    for i in range(len(all_folders)):
        current_run_tif = OS_Tools.Get_File_Name(all_folders[i])
        Before_Align_Tif_Name.append(current_run_tif)
        
    #%% Step2, Generate average map before align.
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
        if len(all_folders)>1:
            for i in range(len(Before_Align_Tif_Name)):
                Graph_Tools.Show_Graph(global_average_graph, 'Global_Average', all_save_folders[i],show_time = 0)
        else:
            print('Only One run, no global average.')
    else:
        print('Before average Skipped.')
    time_tic_average0 = time.time()
    
    #%% Step3, Core Align Function.
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
            All_Aligned_Frame[i] = np.zeros(shape = (graph_shape+(len(Before_Align_Tif_Name[i]),)),dtype = 'u2')# Generate empty graph matrix.   
    for i in range(len(Before_Align_Tif_Name)): # Cycle all runs
        for j in range(len(Before_Align_Tif_Name[i])): # Cycle all graphs in current run
            current_graph = cv2.imread(Before_Align_Tif_Name[i][j],-1) # Read in current graph.
            _,_,current_aligned_graph = Alignment(base, current_graph,boulder = align_boulder,align_range = align_range)
            graph_name = Before_Align_Tif_Name[i][j].split('\\')[-1][:-4] # Ignore extend name'.tif'.
            Graph_Tools.Show_Graph(current_aligned_graph,graph_name,Aligned_frame_folders[i],show_time = 0)
            if big_memory_mode == True:
                All_Aligned_Frame[i][:,:,j] = current_aligned_graph
    print('Align Finished, generating average graphs...')
    time_tic_align_finish = time.time()
    
    #%% Step4, After Align Average
    After_Align_Graphs = {}
    if big_memory_mode == True:# Average can be faster.
        temp_global_average_after_align = np.zeros(shape = graph_shape,dtype = 'f8')
        for i in range(len(All_Aligned_Frame)):
            current_run_average = Graph_Tools.Clip_And_Normalize(np.mean(All_Aligned_Frame[i],axis = 2),clip_std = average_std) # Average run graphs, in type 'u2'
            After_Align_Graphs[i] = (current_run_average,len(All_Aligned_Frame[i][0,0,:]))
            temp_global_average_after_align += After_Align_Graphs[i][0].astype('f8')*After_Align_Graphs[i][1]/total_graph_num
        global_average_after_align = Graph_Tools.Clip_And_Normalize(temp_global_average_after_align,clip_std = average_std)
    else: # Traditional ways.
        temp_global_average_after_align = np.zeros(shape = graph_shape,dtype = 'f8')
        for i in range(len(Aligned_frame_folders)):
            current_run_names = OS_Tools.Get_File_Name(Aligned_frame_folders[i])
            current_run_average = Graph_Tools.Average_From_File(current_run_names)
            current_run_average = Graph_Tools.Clip_And_Normalize(current_run_average,clip_std = average_std)
            After_Align_Graphs[i] = (current_run_average,len(current_run_names))
            temp_global_average_after_align += After_Align_Graphs[i][0].astype('f8')*After_Align_Graphs[i][1]/total_graph_num
        global_average_after_align = Graph_Tools.Clip_And_Normalize(temp_global_average_after_align,clip_std = average_std)
    # After average, save aligned graph in each save folder.
    for i in range(len(all_save_folders)):
        current_save_folder = all_save_folders[i]
        Graph_Tools.Show_Graph(After_Align_Graphs[i][0], 'Run_Average_After_Align', current_save_folder)
        if i == 0:# Show global average only once.
            global_show_time = 5000
        else:
            global_show_time = 0
        if len(all_folders)>1:
            Graph_Tools.Show_Graph(global_average_after_align, 'Global_Average_After_Align', current_save_folder,show_time = global_show_time)
    time_tic_average1 = time.time()
    
    #%% Step5, save and timer
    if save_aligned_data == True:
        OS_Tools.Save_Variable(all_save_folders[0], 'All_Aligned_Frame_Data', All_Aligned_Frame)
        
    if timer == True:
        whole_time = time_tic_average1-time_tic_start
        before_average_time = time_tic_average0-time_tic_start
        align_time = time_tic_align_finish-time_tic_average0
        after_average_time = time_tic_average1-time_tic_align_finish
        print('Total Time = '+str(whole_time)+' s.')
        print('Before Average Time = '+str(before_average_time)+' s.')
        print('Align Time = '+str(align_time)+' s.')
        print('After Average Time = '+str(after_average_time)+' s.')
        
    return True
