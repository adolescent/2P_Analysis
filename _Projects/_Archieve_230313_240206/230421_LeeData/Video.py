'''
Pack video for lee data.
'''
#%%

from tqdm import tqdm
from My_Wheels.Video_Writer import Video_From_mat
import OS_Tools_Kit as ot
import caiman as cm
import numpy as np
import matplotlib.pyplot as plt
import cv2

wp = r'D:\ZR\_Data_Temp\2pt_T151425_A2\_CAIMAN'
filename = ot.Get_File_Name(wp,'.mmap')[0]
Yr, dims, T = cm.load_memmap(filename)
images = np.reshape(Yr.T, [T] + list(dims), order='F')
#%% bin graph into 4 Hz
import pandas as pd
all_cell_frame = pd.DataFrame(Yr.T)
bin4_frame = all_cell_frame.groupby(np.arange(len(all_cell_frame))//8).mean()

#%% gain and clip.
# gain = 20
# fps = 31
# bin = 4
# a = np.clip((images[500:700,:,:]/255)*gain,0,255).astype('u1')
# graph_size = (512,512)
# graph_num = images.shape[0]
# # fourcc = cv2.VideoWriter_fourcc(*'PNG*')
# video_writer = cv2.VideoWriter(wp+r'\\Video.mp4',cv2.VideoWriter_fourcc('X','V','I','D'),fps,graph_size,0)
# for i in tqdm(range(200)):
#     video_writer.write(a[i,:,:])
# del video_writer 
gain = 5
binned_frames = np.array(bin4_frame).reshape(9285,512,512)
binned_frames = np.clip((binned_frames/255)*gain,0,255)

import My_Wheels.Filters as Filters
input_matrix = binned_frames
fps = 12
print('Generate video from file, make sure input file is u1 type.')
graph_size = (input_matrix.shape[1],input_matrix.shape[2])
graph_num = input_matrix.shape[0]
video_writer = cv2.VideoWriter(wp+r'\\Video.mp4',cv2.VideoWriter_fourcc('X','V','I','D'),fps,graph_size,0)
for i in tqdm(range(graph_num)):
    c_graph = input_matrix[i,:,:]
    # u1_writable_graph = c_graph.astype('f8')
    u1_writable_graph = Filters.Filter_2D(c_graph,([5,5],1.5),False)
    u1_writable_graph = u1_writable_graph.astype('u1')
    video_writer.write(u1_writable_graph)
del video_writer 
