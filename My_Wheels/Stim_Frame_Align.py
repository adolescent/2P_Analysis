# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 18:14:59 2020

@author: zhang

Align stim id and frames.
-1 as stim off.
"""

import OS_Tools_Kit as os_tools
import more_itertools as mit
import numpy as np


def Stim_Frame_Align(
        stim_folder,
        stim_thres = 2,
        frame_thres = 1,
        jmp_step = 3000,
        head_extend = 1,
        tail_extend = 0,
        ):
    """
    Get stim belongings of every frame.

    Parameters
    ----------
    stim_folder : (str)
        Stimlus data folder. '.smr' file and '.txt' file shall be in the same folder.
    stim_thres :(number),optional
        Threshold voltage used to binary square wave. The default is 2.
    frame_thres:(number),optional
        Threshold voltage used to binary triangel wave. The default is 1.
    jmp_step:(int),optional
        How many point you jump after find a frame. Usually, 10000 point = 1s
    head_extend(int),optional
        Number of frame regarded as stim on before stim. Positive will extend frame on, Negative will cut.
    tail_extend(int),optional
        Number of frame ragarded as stim on after stim. Positive will extend frame on, Negative will cut.
    Returns
    -------
    Frame_Stim_Sequence : (list)
        List type of frame belongings. This can be used if ISI base changes.
    Frame_Stim_Dictionary : (Dictionary)
        Dictionary type. This Dictionary have stim id belonged frames. Can be used directly.
    """
    # Step 1, read in data.
    smr_name = os_tools.Get_File_Name(stim_folder,file_type = '.smr')[0]
    frame_train = os_tools.Spike2_Reader(smr_name,stream_channel ='0')['Channel_Data']
    stim_train = os_tools.Spike2_Reader(smr_name,stream_channel ='1')['Channel_Data']
    txt_name = os_tools.Last_Saved_name(stim_folder,file_type = '.txt')
    
    # Step 2, square wave series processing
    binary_stim_train = (stim_train>stim_thres).astype('i4')
    cutted_stim_list = list(mit.split_when(binary_stim_train,lambda x, y: (x-y) == -1))
    # If stop at high voltage level, change last square to -1.
    last_part_set = np.unique(cutted_stim_list[-1])
    if len(last_part_set) == 1: # Which means stop at high voltage
        last_part = np.array(cutted_stim_list[-1])
        last_part[:] = 0
        cutted_stim_list[-1] = list(last_part)
    # Combine stimuls lists
    final_stim_list = []
    for i in range(len(cutted_stim_list)):
        current_list = np.dot(cutted_stim_list[i],i+1)-1
        final_stim_list.extend(current_list)
    del cutted_stim_list,stim_train,binary_stim_train
    # square wave process done, final_stim_list is stim-time relation.
    # Step3, triangle wave list processing.
    binary_frame_train = (frame_train>frame_thres).astype('i4').ravel()
    dislocation_binary_frame_train = np.append(binary_frame_train[1:],0)
    frame_time_finder = binary_frame_train - dislocation_binary_frame_train
    stop_point = np.where(frame_time_finder == -1)[0] # Not filtered yet, mis calculation are many.
    # Wash stop points, make sure they have 
    all_graph_time = [stop_point[0]]# Use first stop as first graph.
    last_frame_time = all_graph_time[0]# First stop
    for i in range(1,len(stop_point)):# Frame 0 ignored.
        current_time = stop_point[i]
        if (current_time - last_frame_time)>jmp_step:
            all_graph_time.append(current_time)
            last_frame_time = current_time
    all_graph_time = all_graph_time[:-2] # Last 2 frame may not be saved.
    # Triangle wave process done, all_graph_time is list of every frame time.
    
    # Step4,Original frame stim relation acquire.
    frame_belongings = []
    for i in range(len(all_graph_time)):
        current_graph_time = all_graph_time[i]
        frame_belongings.append(final_stim_list[current_graph_time][0])# Frame belong before adjust
    
    # Step5, Adjust frame stim relation.
    cutted_frame_list = list(mit.split_when(frame_belongings,lambda x, y: x!=y))
    # Adjust every single part. Stim add means ISI subs.
    adjusted_frame_list = []
    import My_Wheels.List_Operation_Kit as List_Ops
    # Process head first
    adjusted_frame_list.append(List_Ops.List_extend(cutted_frame_list[0],0,-head_extend))
    # Then Process middle
    for i in range(1,len(cutted_frame_list)-1):# First and last frame use differently.
        if (i%2) != 0:# odd id means stim on.
            adjusted_frame_list.append(List_Ops.List_extend(cutted_frame_list[i],head_extend,tail_extend))
        else:# even id means ISI.
            adjusted_frame_list.append(List_Ops.List_extend(cutted_frame_list[i],-tail_extend,-head_extend))
    # Process last part then.
    adjusted_frame_list.append(List_Ops.List_extend(cutted_frame_list[-1],-tail_extend,0))
    # After adjustion, we need to combine the list.
    frame_stim_list = []
    for i in range(len(adjusted_frame_list)-1):# Ignore last ISI, this might be harmful.
        frame_stim_list.extend(adjusted_frame_list[i])
    # Till now, frame_stim_list is adjusted frame stim relations.
    
    # Step6, Combine frame with stim id.
    with open(txt_name,'r') as file:
        data = file.read()
    del file
    stim_sequence = data.split()
    stim_sequence = [int(x) for x in stim_sequence]
    Frame_Stim_Sequence = []
    for i in range(len(frame_stim_list)):
        current_id = frame_stim_list[i]
        if current_id != -1:
            Frame_Stim_Sequence.append(stim_sequence[current_id-1])
        else:
            Frame_Stim_Sequence.append(-1)
    Frame_Stim_Dictionary = List_Ops.List_To_Dic(Frame_Stim_Sequence)
    Frame_Stim_Dictionary['Original_Stim_Train'] = Frame_Stim_Sequence
    return Frame_Stim_Sequence,Frame_Stim_Dictionary


#%% Define a function to calculate all aligns in a single day.
def One_Key_Stim_Align(
        stims_folder
        ):
    '''
    Generate all stim trains 

    Parameters
    ----------
    stims_folder : (str)
        Folder of all stim files.

    Returns
    -------
    bool
        API of operation finish.

    '''
    all_stim_folders = os_tools.Get_Sub_Folders(stims_folder)
    # remove folders not stim run
    for i in range(len(all_stim_folders)-1,-1,-1):
        if all_stim_folders[i].find('Run') == -1:
            all_stim_folders = np.delete(all_stim_folders, i)
    total_save_path = os_tools.CDdotdot(stims_folder)
    All_Stim_Dic = {}
    # Then generate all align folders.
    for i in range(len(all_stim_folders)):
        current_stim_folder = all_stim_folders[i]
        current_runname = list(current_stim_folder[current_stim_folder.find('Run'):current_stim_folder.find('Run')+5])
        current_runname.insert(3,'0')
        current_runname = ''.join(current_runname)
        not_spon = (os_tools.Get_File_Name(current_stim_folder,file_type = '.smr') != [])
        if not_spon:
            _,current_align_dic = Stim_Frame_Align(current_stim_folder)
            os_tools.Save_Variable(current_stim_folder, 'Stim_Frame_Align', current_align_dic)
            All_Stim_Dic[current_runname] = current_align_dic
        else:
            All_Stim_Dic[current_runname] = None
    os_tools.Save_Variable(total_save_path, '_All_Stim_Frame_Infos', All_Stim_Dic,'.sfa')
    return True