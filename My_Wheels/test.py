# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 16:25:07 2020

@author: ZR
Test Run File
Do not save.
"""
#%%
import My_Wheels.OS_Tools_Kit as os_tools
import numpy as np


work_folder = r'E:\Test_Data\200107_L80_LM\200107_L80_2P_stimuli\Run01_2P_G8'
smr_name = os_tools.Get_File_Name(work_folder,file_type = '.smr')[0]
frame_train = os_tools.Spike2_Reader(smr_name,physical_channel = 3)['Channel_Data']
stim_train = os_tools.Spike2_Reader(smr_name,physical_channel = 0)['Channel_Data']
#%% Read last saved txt file.
txt_name = os_tools.Last_Saved_name(work_folder,file_type = '.txt')

#%% Process stim train first
import more_itertools as mit
stim_thres = 2
binary_stim_train = (stim_train>stim_thres).astype('i4')
cutted_stim_list = list(mit.split_when(binary_stim_train,lambda x, y: (x-y) == -1))
# Zero last stim if stop at high voltage.

last_part_set = np.unique(cutted_stim_list[-1])
if len(last_part_set) == 1: # Which means stop at high voltage
    last_part = np.array(cutted_stim_list[-1])
    last_part[:] = -1
    cutted_stim_list[-1] = list(last_part)
# Last, combine cutted series
processed_list = []
for i in range(len(cutted_stim_list)):
    current_list = np.dot(cutted_stim_list[i],i+1)-1
    processed_list.extend(current_list)
# Till now, processed_list is the wanted list. -1 for no stim, n as stim number.
del cutted_stim_list,stim_train,binary_stim_train
#%% This part will process frame data. last frame ignored.
frame_thres = 1
jmp_step = 3000 # Least frame dist, usually 10000 as 1s
binary_frame_train = (frame_train>frame_thres).astype('i4').ravel()
dislocation_binary_frame_train = np.append(binary_frame_train[1:],0)
frame_time_finder = binary_frame_train - dislocation_binary_frame_train
stop_point = np.where(frame_time_finder == -1)[0]

all_graph_time = [stop_point[0]]# Use first stop as first graph.
last_frame_time = all_graph_time[0]# First stop
for i in range(1,len(stop_point)):# Frame 0 ignored.
    current_time = stop_point[i]
    if (current_time - last_frame_time)>jmp_step:
        all_graph_time.append(current_time)
        last_frame_time = current_time
all_graph_time = all_graph_time[:-2] # Last 2 frame may not be saved.
# Till now, all_graph_time contains all frame times.
#%% This part will get frame-stim sequence
frame_belongings = []
for i in range(len(all_graph_time)):
    current_graph_time = all_graph_time[i]
    frame_belongings.append(processed_list[current_graph_time][0])# Frame belong before adjust
#%% Adjust stim-belonged frame.
head_extend = -1 # how many frame before stim show added
tail_extend = 0 # how many frame after stim show added
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
for i in range(len(adjusted_frame_list)):
    frame_stim_list.extend(adjusted_frame_list[i])
# Till now, frame_stim_list is the adjusted frame-stim relations.
#%% Read txt, get stim sequence now.
with open(txt_name,'r') as file:
    data = file.read()
del file
stim_sequence = data.split()
stim_sequence = [int(x) for x in stim_sequence]# change into int type.
#%% This is the last part. distribute stim into stim id.
Frame_Stim_Sequence = []
for i in range(len(frame_stim_list)):
    current_id = frame_stim_list[i]
    if current_id != -1:
        Frame_Stim_Sequence.append(stim_sequence[current_id-1])
    else:
        Frame_Stim_Sequence.append(-1)
Frame_Stim_Dictionary = List_Ops.List_To_Dic(Frame_Stim_Sequence)