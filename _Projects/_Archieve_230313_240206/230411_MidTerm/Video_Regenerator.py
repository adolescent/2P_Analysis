'''
Created by ZR in 230330, this provide new results for mid-term report.
'''

#%% import first

import OS_Tools_Kit as ot
import pandas as pd
from tqdm import tqdm
from Filters import Signal_Filter
import Graph_Operation_Kit as gt
day_folder = r'D:\ZR\_Temp_Data\220630_L76_2P'

#%% Regenerate L76-0630 aligned graphs, with Run01 4000 frame cutted.
from Caiman_API.Precess_Pipeline import Preprocess_Pipeline
day_folder = r'D:\ZR\_Temp_Data\220630_L76_2P'
pp = Preprocess_Pipeline(day_folder,runlist = [1,3,6,7,8],orien_run='Run007',od_run='Run006',color_run = 'Run008')
pp.Do_Preprocess()

#%% Video Write. First, only cell videos.
# We use 3500-4000 frame as example, generate short video cuts.
from Series_Analyzer.Preprocessor_Cai import Pre_Processor_Cai
import numpy as np
import cv2
run01_frames = Pre_Processor_Cai(day_folder,'Run001')
acd = ot.Load_Variable(r'D:\ZR\_Temp_Data\220630_L76_2P\_CAIMAN\All_Series_Dic.pkl')
all_cell_loc = {}
for i,cc in tqdm(enumerate(run01_frames.index)):
    cc_loc = acd[cc]['Cell_Loc']
    ccx = int(cc_loc[1])
    ccy = int(cc_loc[0])
    all_cell_loc[cc] = (ccx,ccy)
plotable_parts = run01_frames.loc[:,3500:4000]
# threshold, clip and normalize
thresed_graph = plotable_parts*(plotable_parts>2)
thresed_graph = thresed_graph.T
clipped_graph = thresed_graph.clip(upper = 5)
normed_graph = clipped_graph/clipped_graph.max() # this is 0-1 graphs.
cell_num = normed_graph.shape[1]
# generate matrix so we can plot.
graph_tiles = np.zeros(shape = (512,512,501),dtype = 'u1')
for i in tqdm(range(501)):# draw all graph
    c_frame = np.zeros(shape = (512,512),dtype = 'u1')
    for j in range(cell_num):
        cc_weight = normed_graph.iloc[i,j]*255
        ccx,ccy = all_cell_loc[j+1]
        c_frame = cv2.circle(c_frame,(ccx,ccy),5,cc_weight,-1)
    graph_tiles[:,:,i] = c_frame

# Write here.
from Video_Writer import Video_From_mat
save_path = r'D:\ZR\_Temp_Data\220630_L76_2P\_CAIMAN'
Video_From_mat(graph_tiles,save_path,fps = 8)
# %% get frame video here.
import caiman as cm
filename = r'D:\ZR\_Temp_Data\220630_L76_2P\_CAIMAN\1-001_0000-#-5-#-5_d1_512_d2_512_d3_1_order_C_frames_23668_.mmap'
Y,dims,T = cm.load_memmap(filename)
used_Y = Y[:,3500:4000]
used_frames = np.reshape(used_Y.T, [500,512,512], order='F')
# used_frames = Y[:,3500:4000].reshape(dims[0],dims[1],-1)
# del Y,dims,T
# gain graph
import Filters
gained_graph = np.clip(used_frames.astype('f8')*20/256,0,255).astype('u1')
img_tiles = np.zeros(shape = (512,512,500),dtype = 'u1')
for i in tqdm(range(500)):
    u1_writable_graph = Filters.Filter_2D(gained_graph[i,:,:],LP_Para = ([5,5],1.5),HP_Para = False)
    cv2.putText(u1_writable_graph,'Stim ID = '+str(i),(250,30),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255),1)
    img_tiles[:,:,i] = u1_writable_graph
Video_From_mat(img_tiles,save_path,fps = 8)
#%% Tile 2 videos together.
tiled_graph = np.zeros(shape = (512,1024,500),dtype = 'u1')
for i in tqdm(range(500)):
    stacked = np.hstack((graph_tiles[:,:,i],img_tiles[:,:,i]))
    tiled_graph[:,:,i] = stacked
Video_From_mat(tiled_graph,save_path,fps = 8)
#%% big,small and local
big_ensemble = graph_tiles[:,:,420]
small_ensemble = graph_tiles[:,:,286]
local_ensemble = graph_tiles[:,:,369]
gt.Show_Graph(big_ensemble,'Big_ensemble',save_path)
gt.Show_Graph(small_ensemble,'Small_ensemble',save_path)
gt.Show_Graph(local_ensemble,'Local_ensemble',save_path)
