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

smr_path = r'E:\Test_Data\200107_L80_LM\200107_L80_2P_stimuli\Run01_2P_G8'
smr_name = '1.smr'
frame_train = os_tools.Spike2_Reader(smr_name,smr_path,physical_channel = 3)['Channel_Data']
stim_train = os_tools.Spike2_Reader(smr_name,smr_path,physical_channel = 0)['Channel_Data']
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
#%% This part will get frame-stim sequence, we need to adjust sequence for convenience.
head_extend = 1 # how many frame before stim show added
tail_extend = 1 # how many frame after stim show added

frame_belongings = []
for i in range(len(all_graph_time)):
    current_graph_time = all_graph_time[i]
    frame_belongings.append(processed_list[current_graph_time][0])# Frame belong before adjust
    
#%% This part will distribute frame to stim or stim off.
#%% This is the last part. distribute stim into stim id.
