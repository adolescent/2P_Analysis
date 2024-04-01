'''
This script add stim on locations of videos, and try to clip and cut F value videos.



'''
#%%
from Cell_Class.Stim_Calculators import Stim_Cells
from Cell_Class.Format_Cell import Cell
import OS_Tools_Kit as ot
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import cv2
import warnings
from Filters import Signal_Filter_v2

warnings.filterwarnings("ignore")

wp = r'D:\_All_Spon_Data_V1\L76_18M_220902'
ac = ot.Load_Variable(wp,'Cell_Class.pkl')
spon_series = ot.Load_Variable(wp,'Spon_Before.pkl')


# if we need raw frame dF values
raw_spon_run = ot.Load_Variable(f'{wp}\\Spon_Before_Raw.pkl')
spon_avr = raw_spon_run.mean(0)
spon_std = raw_spon_run.std()
used_spon = raw_spon_run[5140:5220,:,:]
del raw_spon_run
raw_orien_run = ot.Load_Variable(f'{wp}\\Orien_Frames_Raw.pkl')
orien_avr = raw_orien_run.mean(0)
orien_std = raw_orien_run.std()
used_orien = raw_orien_run[1140:1220,:,:]
del raw_orien_run
#%% get comparing dff matrix and stim ids.
def dFF(F_series,method = 'least',prop=0.1): # dFF method can be changed here.
    if method == 'least':
        base_num = int(len(F_series)*prop)
        base_id = np.argpartition(F_series, base_num)[:base_num]
        base = F_series[base_id].mean()
    dff_series = (F_series-base)/base
    return dff_series,base

def Generate_F_Series(ac,runname = '1-001',start_time = 0,stop_time = 999999,HP = 0.005,LP = 0.3):
    acd = ac.all_cell_dic
    acn = ac.acn
    
    stop_time = min(len(acd[1][runname]),stop_time)
    # get all F frames first
    F_frames_all = np.zeros(shape = (stop_time-start_time,len(acn)),dtype='f8')
    for j,cc in enumerate(acn):
        c_series_raw = acd[cc][runname][start_time:stop_time]
        # c_series_all = Signal_Filter(c_series_raw,order=5,filter_para=filter_para)
        c_series_all = Signal_Filter_v2(c_series_raw,order=5,HP_freq=HP,LP_freq=LP,fps=1.301)
        F_frames_all[:,j] = c_series_all
    # then cut ON parts if needed.
    output_series = F_frames_all
    return output_series

def dFF_Matrix(F_matrix,method = 'least',prop=0.1):
    dFF_Matrix = np.zeros(shape = F_matrix.shape,dtype = 'f8')
    for i in range(F_matrix.shape[1]):
        c_F_series = F_matrix[:,i]
        c_dff_series,_ = dFF(c_F_series,method,prop)
        dFF_Matrix[:,i] = c_dff_series
    return dFF_Matrix

#%%#################  Define Grating Function #################
from scipy import ndimage
cycle_pix = 20
duty_cycle = 0.2
grating_angle = 30
graph_size = 200
def Generate_Angle_Mask(graph_size = 200,duty_cycle = 0.2,
                        cycle_pix = 25,grating_angle = 30
                        ):
# generate orien 0 matrixs.
    on_len = int(cycle_pix*duty_cycle)
    # off_len = cycle_pix-on_len
    cycle_num = int((graph_size*1.414//cycle_pix)+1)
    fullgraph = np.zeros(shape = (int(cycle_num*cycle_pix),int(cycle_num*cycle_pix)),dtype='u1')
    # rotate any angled graphs.
    for i in range(cycle_num):
        fullgraph[cycle_pix*i:cycle_pix*i+on_len,:] = 1
    rotated_graph = ndimage.rotate(fullgraph, grating_angle,reshape=False)
    # cut the mask into target size.
    pix_range = (len(rotated_graph)//2-graph_size//2,len(rotated_graph)//2+graph_size//2)
    cutted_graph = rotated_graph[pix_range[0]:pix_range[1],pix_range[0]:pix_range[1]]
    # make a circled mask.
    mask = np.zeros(cutted_graph.shape, dtype=np.uint8)
    cv2.circle(mask, (cutted_graph.shape[0]//2,cutted_graph.shape[1]//2), cutted_graph.shape[0]//2, 1, thickness=cv2.FILLED)
    # plt.imshow(rotated_graph)
    masked_graph = cutted_graph*mask
    # plt.imshow(masked_graph)
    return masked_graph

#%%################## PLOT STIM CELL WITH FRAME DATA.#####
# 1. generate cell dff series
g16_stim_response = Generate_F_Series(ac,ac.orienrun)
g16_dff = dFF_Matrix(g16_stim_response)
used_g16_dff = g16_dff[1140:1220,:]
g16_dff_matrix = np.zeros(shape = (len(used_g16_dff),512,512))
for i in range(len(used_g16_dff)):
    g16_dff_matrix[i,::,:] = ac.Generate_Weighted_Cell(used_g16_dff[i,:])

spon_response = Generate_F_Series(ac,'1-001')
spon_dff = dFF_Matrix(spon_response)
used_spon_dff = spon_dff[8500+5140:8500+5220,:]
spon_dff_matrix = np.zeros(shape = (len(used_spon_dff),512,512))
for i in range(len(used_spon_dff)):
    spon_dff_matrix[i,::,:] = ac.Generate_Weighted_Cell(used_spon_dff[i,:])

#%% clip and normalize for frame data and cell data.
# clip_std = 25
frame_clip = 4000
dff_clip = 3
# clipped_spon_raw = np.clip(used_spon,spon_avr.mean()-spon_std*clip_std,spon_avr.mean()+spon_std*clip_std)
clipped_spon_raw = np.clip(used_spon,0,frame_clip)
# clipped_orien_raw = np.clip(used_orien,orien_avr.mean()-orien_std*clip_std,orien_avr.mean()+orien_std*clip_std)
clipped_orien_raw = np.clip(used_orien,0,frame_clip)


# clipped_orien_cell = np.clip(g16_dff_matrix,g16_dff_matrix.mean()-g16_dff_matrix.std()*clip_std,g16_dff_matrix.mean()+g16_dff_matrix.std()*clip_std)
clipped_orien_cell = np.clip(g16_dff_matrix,-1,dff_clip)
# clipped_spon_cell = np.clip(spon_dff_matrix,spon_dff_matrix.mean()-spon_dff_matrix.std()*clip_std,spon_dff_matrix.mean()+spon_dff_matrix.std()*clip_std)
clipped_spon_cell = np.clip(spon_dff_matrix,-1,dff_clip)


# normalize and concat graphs together.
clipped_orien_raw = clipped_orien_raw[:,20:492,20:492]
clipped_orien_cell = clipped_orien_cell[:,20:492,20:492]
clipped_spon_raw = clipped_spon_raw[:,20:492,20:492]
clipped_spon_cell = clipped_spon_cell[:,20:492,20:492]

plotable_orien = np.concatenate((clipped_orien_raw/clipped_orien_raw.max(),clipped_orien_cell/clipped_orien_cell.max()),axis=2)
plotable_spon = np.concatenate((clipped_spon_raw/clipped_spon_raw.max(),clipped_spon_cell/clipped_spon_cell.max()),axis=2)
# Add stim ids into the orien parts.
stim_ids = np.array(ac.Stim_Frame_Align['Run007']['Original_Stim_Train'][1140:1220])
stim_shows = np.zeros(shape = (80,472,100),dtype='f8')
for i,c_id in enumerate(stim_ids):
    if c_id == -1 or c_id == 0:
        real_id = -1
    elif c_id>8:
        real_id = c_id-8
    else:
        real_id = c_id
    if real_id>0:
        real_orien = (real_id-1)*22.5
        c_stim = Generate_Angle_Mask(100,0.2,25,real_orien)
        stim_shows[i,:100,:] = c_stim
    
plotable_orien = np.concatenate((plotable_orien,stim_shows),axis=2)
#%%
# frames = used_spon*255/used_spon.max()
# frames = clipped_spon_cell*255/clipped_spon_cell.max()
frames = plotable_orien*255
# frames = plotable_spon*255

import cv2
import skvideo.io
import time
width = 1124
height = 512
fps = 4
outputfile = "test.avi"   #our output filename
writer = skvideo.io.FFmpegWriter(outputfile, outputdict={
    '-vcodec': 'rawvideo',  #use the h.264 codec
    #  '-vcodec': 'libx264',
    '-crf': '0',           #set the constant rate factor to 0, which is lossless
    '-preset':'veryslow',   #the slower the better compression, in princple, 
    # '-r':str(fps), # this only control the output frame rate.
    '-pix_fmt': 'yuv420p',
    '-vf': "setpts=PTS*{},fps={}".format(25/fps,fps) ,
    '-s':'{}x{}'.format(width,height)
}) 

for frame in tqdm(frames):
    # cv2.imshow('display',frame)
    writer.writeFrame(frame)  #write the frame as RGB not BGR
    # time.sleep(1/fps)

writer.close() #close the writer
cv2.destroyAllWindows()

