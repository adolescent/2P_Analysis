# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 13:42:28 2020

@author: zhang

Generate spike train
"""
import cv2
import numpy as np
import more_itertools as mit
import My_Wheels.List_Operation_Kit as List_Tools
import My_Wheels.Filters as My_Filter


def Spike_Train_Generator(all_tif_name,
                          cell_information,
                          Base_F_type = 'most_unactive',
                          stim_train = None,
                          ignore_ISI_frame = 1,
                          unactive_prop = 0.1,
                          LP_Para = False,
                          HP_Para = False,
                          filter_method = False
                          ):
    """
    
    Generate Spike Train from graphs. Multiple find method provided.
    Filter here indicating 2D spacial filter. No time course filter provided here.
    
    Parameters
    ----------
    all_tif_name : (list)
        List of all tif graph.
    cell_information : (list)
        Skimage generated cell information lists.
    Base_F_type : ('global','most_unactive','last_ISI','begining_ISI','all_ISI','nearest_0','all_0'), optional
        Base F find method. Describtion as below:
            'global' : Use all frame average.
            'most_unactive': Use most lazy frames of every cell.
            'before_ISI': Use the very ISI before stim onset as base. 
            'begining_ISI': Use ISI before stim onset as base.
            'all_ISI': Use average of all ISI as base. Each ISI will be cut based on ignore_ISI_frame.
            'nearest_0': Use nearest stim id 0 as base.
            'all_0': Use average of all id 0 data as base.
        The default is 'global'.
    stim_train : (list), optional
        Stim id train. If Base type include stim information, this must be given. The default is None.
    ignore_ISI_frame : TYPE, optional
        For mode 'last_ISI'/'all_ISI'/'begining_ISI'. How much ISI fram will be ingored. The default is 1.
    unactive_prop : (float), optional
        For mode 'most_unactive'. Propotion of most unactive frame used. The default is 0.1.

    Returns
    -------
    dF_F_trains : (Dictionary)
        Spike train of every cell. Note there is only spike train, submap will be processed later.
    F_value_Dictionary : (Dictionary)
        Origional F value dictionary.
    
    """
    # Initialization
    Cell_Num = len(cell_information)
    Frame_Num = len(all_tif_name)
    F_value_Dictionary = {}
    height,width = np.shape(cv2.imread(all_tif_name[0],-1))
    all_graph_matrix = np.zeros(shape = (height,width,Frame_Num),dtype = 'u2')
    # Step 1, read in all graphs. Do filter is required.
    for i in range(Frame_Num):
        current_graph = cv2.imread(all_tif_name[i],-1)
        if filter_method != False: # Meaning we need filter here.
            current_graph = My_Filter.Filter_2D(current_graph,LP_Para,HP_Para,filter_method)
        all_graph_matrix[:,:,i] = current_graph
        

    # Step 2, generate origin F value list first.
    for i in range(Cell_Num):# cycle cell
        cell_location = cell_information[i].coords
        cell_area = len(cell_location)
        current_cell_train = all_graph_matrix[cell_location[:,0],cell_location[:,1],:].astype('f8')
        current_cell_F_train = np.sum(current_cell_train,axis = 0)/cell_area
        F_value_Dictionary[i] = current_cell_F_train
    del all_graph_matrix
    # Step3, after getting F Dictionary, it's time to calculate dF/F matrix.
    dF_F_trains  = {}
    all_keys = list(F_value_Dictionary.keys())
    
    if Base_F_type == 'global':
        for i in range(len(all_keys)):
            current_cell_F_train = F_value_Dictionary[all_keys[i]]
            base_F = current_cell_F_train.mean()
            current_spike_train = (current_cell_F_train-base_F)/base_F
            dF_F_trains[all_keys[i]] = current_spike_train
            
    elif Base_F_type == 'most_unactive':
        for i in range(len(all_keys)):
            current_cell_F_train = F_value_Dictionary[all_keys[i]]
            # Base is avr. of most unactive frames.
            sorted_list = sorted(current_cell_F_train)# Use this to get mean.
            unactive_frame_num = round(len(sorted_list)*unactive_prop)
            sorted_list = sorted_list[:unactive_frame_num]
            base_F = np.mean(sorted_list)
            current_spike_train = (current_cell_F_train-base_F)/base_F
            dF_F_trains[all_keys[i]] = current_spike_train
            
    elif Base_F_type == 'before_ISI':# Use ISI Before stim onset as base.
        if stim_train == None:
            raise IOError('Please input stim train!')
        stim_train = np.asarray(stim_train)
        ignore_ISI_frame = 1
        all_keys = list(F_value_Dictionary.keys())
        cutted_stim_train = list(mit.split_when(stim_train,lambda x, y: (x-y) >0))
        for i in range(len(all_keys)):
            current_cell_train = F_value_Dictionary[all_keys[i]]
            frame_counter = 0
            current_cell_dF_train = []
            for j in range(len(cutted_stim_train)):
                current_stim_train = np.asarray(cutted_stim_train[j])
                current_F_train = np.asarray(current_cell_train[frame_counter:(frame_counter+len(current_stim_train))])
                null_id = np.where(current_stim_train == -1)[0]
                null_id = null_id[ignore_ISI_frame:]
                current_base = current_F_train[null_id].mean()
                current_dF_train = (current_F_train-current_base)/current_base
                current_cell_dF_train.extend(current_dF_train)
                # Then add frame counter at last.
                frame_counter = frame_counter + len(cutted_stim_train[j])
            dF_F_trains[all_keys[i]] = np.asarray(current_cell_dF_train)
            
    elif Base_F_type == 'begining_ISI':# Use First ISI as global base.
        if stim_train == None:
            raise IOError('Please input stim train!')
        first_stim_id = np.where(np.asarray(stim_train)>0)[0][0]
        all_keys = list(F_value_Dictionary.keys())
        for i in range(len(all_keys)):
            current_F_series = F_value_Dictionary[all_keys[i]]
            base_F_series = current_F_series[ignore_ISI_frame:first_stim_id]
            base_F = base_F_series.mean()
            current_spike_train = (current_F_series-base_F)/base_F
            dF_F_trains[all_keys[i]] = current_spike_train
            
    elif Base_F_type == 'all_ISI':
        if stim_train == None:
            raise IOError('Please input stim train!')
        stim_train = np.asarray(stim_train)
        all_ISI_frame_loc = np.where(stim_train == -1)[0]
        cutted_ISI_frame_loc = list(mit.split_when(all_ISI_frame_loc,lambda x,y:(y-x)>1))
        used_ISI_id = []
        for i in range(len(cutted_ISI_frame_loc)):
            used_ISI_id.extend(cutted_ISI_frame_loc[i][ignore_ISI_frame:])
        all_keys = list(F_value_Dictionary.keys())
        for i in range(len(all_keys)):
            current_cell_F_train = F_value_Dictionary[all_keys[i]]
            current_base_F = current_cell_F_train[used_ISI_id]
            base_F = current_base_F.mean()
            current_dF_train = (current_cell_F_train-base_F)/base_F
            dF_F_trains[all_keys[i]] = current_dF_train

    elif Base_F_type == 'nearest_0':
        stim_train = np.asarray(stim_train)
        blank_location = np.where(stim_train == 0)[0]
        cutted_blank_location = list(mit.split_when(blank_location,lambda x,y:(y-x)>1))
        all_blank_start_frame = [] # This is the start frame of every blank.
        for i in range(len(cutted_blank_location)):
            all_blank_start_frame.append(cutted_blank_location[i][0])
        #%% Get base_F_of every blank.
        all_keys = list(F_value_Dictionary.keys())
        for i in range(len(all_keys)):
            current_key = all_keys[i]
            current_cell_F_train = F_value_Dictionary[current_key]
            # First, get base F of every blank.
            all_blank_base_F = [] # base F of every blank.
            for j in range(len(cutted_blank_location)):
                all_blank_base_F.append(current_cell_F_train[cutted_blank_location[j]].mean())
            # Then, generate dF train.
            current_dF_train = []
            for j in range(len(current_cell_F_train)):
                current_F = current_cell_F_train[j]
                _,current_base_loc = List_Tools.Find_Nearest(all_blank_start_frame,j)
                current_base = all_blank_base_F[current_base_loc]
                current_dF_F = (current_F-current_base)/current_base
                current_dF_train.append(current_dF_F)
            dF_F_trains[all_keys[i]] = np.asarray(current_dF_train)
            
    elif Base_F_type == 'all_0':
        stim_train = np.asarray(stim_train)
        all_blank_frame_id = np.where(stim_train == 0)[0]
        all_keys = list(F_value_Dictionary.keys())
        for i in range(len(all_keys)):
            current_cell_F_train = F_value_Dictionary[all_keys[i]]
            current_base = current_cell_F_train[all_blank_frame_id].mean()
            current_dF_train = (current_cell_F_train-current_base)/current_base
            dF_F_trains[all_keys[i]] = current_dF_train
    
    else:
        raise IOError('Not finished functions.')
    
    return F_value_Dictionary,dF_F_trains
#  return F_value_Dictionary,dF_F_trains

