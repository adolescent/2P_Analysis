'''
This script will generate several videos for stimulus and spontaneous frames.

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

warnings.filterwarnings("ignore")

wp = r'D:\_All_Spon_Data_V1\L76_18M_220902'
ac = ot.Load_Variable(wp,'Cell_Class.pkl')
spon_series = ot.Load_Variable(wp,'Spon_Before.pkl')
# if we need raw frame dF values
raw_orien_run = ot.Load_Variable(f'{wp}\\Orien_Frames_Raw.pkl')
raw_spon_run = ot.Load_Variable(f'{wp}\\Spon_Before_Raw.pkl')

#%% ###################### Type1, Plot STIM HERE.#################
raw_orien_df = raw_orien_run-raw_orien_run.mean(0)
clip_std = 3
raw_orien_df = np.clip(raw_orien_df,-raw_orien_df.std()*clip_std,raw_orien_df.std()*clip_std)
plotable_orien_df = raw_orien_run[1140:1220,:,:]
#%% normalize and plot parts.
array = plotable_orien_df
normalized_array = (array - np.min(array)) / (np.max(array) - np.min(array))
normalized_array = (normalized_array * 255).astype('u1')
# Step 2: Create a list of frames
frames = [frame for frame in normalized_array]
# Step 3: Set up the video writer
# fourcc = cv2.VideoWriter_fourcc(*'h264')
# # fourcc = -1
# fps = 8
# output_file = 'output.mp4'
# video_writer = cv2.VideoWriter(output_file, fourcc, fps, (512,512), isColor=False)
# # Step 4: Write the frames to the video
# for frame in tqdm(frames):
#     video_writer.write(frame)
#     # cv2.imshow('Test',frame)

# # Step 5: Release the video writer and destroy any remaining windows
# video_writer.release()
# cv2.destroyAllWindows()
#%% loseless save using skvideo.
import cv2
import skvideo.io
import time

width = 512
height = 512
fps = 8
outputfile = "test.avi"   #our output filename
writer = skvideo.io.FFmpegWriter(outputfile, outputdict={
    '-vcodec': 'rawvideo',  #use the h.264 codec
    '-crf': '0',           #set the constant rate factor to 0, which is lossless
    '-preset':'veryslow',   #the slower the better compression, in princple, 
    # '-r':str(fps), # this only control the output frame rate.
    '-pix_fmt': 'yuv420p',
    '-vf': "setpts=PTS*3,fps=8" ,
    '-s':'{}x{}'.format(width,height)
}) 

for frame in tqdm(frames):
    # cv2.imshow('display',frame)
    writer.writeFrame(frame)  #write the frame as RGB not BGR
    # time.sleep(1/fps)

writer.close() #close the writer
cv2.destroyAllWindows()
#%%############### PLOT SPON HERE.
raw_spon_df = raw_spon_run-raw_spon_run.mean(0)
clip_std = 3
raw_spon_df = np.clip(raw_spon_df,-raw_spon_df.std()*clip_std,raw_spon_df.std()*clip_std)
plotable_spon_df = raw_spon_run[5140:5220,:,:]
array = plotable_spon_df
normalized_array = (array - np.min(array)) / (np.max(array) - np.min(array))
normalized_array = (normalized_array * 255).astype('u1')
# Step 2: Create a list of frames
frames = [frame for frame in normalized_array]
#%%
width = 512
height = 512
fps = 8
outputfile = "test.mp4"   #our output filename
writer = skvideo.io.FFmpegWriter(outputfile, outputdict={
    '-vcodec': 'libx264',  #use raw video, 'libx264'is better,but might not playable.
    '-crf': '0',           #set the constant rate factor to 0, which is lossless
    '-preset':'veryslow',   #the slower the better compression, in princple, 
    # '-r':str(fps), # this only control the output frame rate.
    '-pix_fmt': 'yuv420p',
    '-vf': "setpts=PTS*3,fps=8" ,
    '-s':'{}x{}'.format(width,height)
}) 

for frame in tqdm(frames):
    # cv2.imshow('display',frame)
    writer.writeFrame(frame)  #write the frame as RGB not BGR
    # time.sleep(1/fps)

writer.close() #close the writer
cv2.destroyAllWindows()



