'''
This part of code will try to transfer data into caimanable type.
First transform is already done in codes of given folder, here we only need to transfer them into tif stashes.

'''
#%%


import h5py
import numpy as np
from My_Wheels.Caiman_API.One_Key_Caiman import One_Key_Caiman
import My_Wheels.OS_Tools_Kit as ot
import numpy as np
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
import tifffile as tif
from tqdm import tqdm
from Cell_Class.Format_Cell import Cell
from Filters import Filter_2D
import caiman as cm
import scipy.io


# plt.switch_backend('webAgg')
day_folder = r'D:\#Lee_Data\231201_Lee_Data\231219_Lee_Data_31fps'
trans_folder = r'D:\#Lee_Data\231201_Lee_Data\231219_Lee_Data_31fps\Transfered'
tif_folder = r'D:\#Lee_Data\231201_Lee_Data\231219_Lee_Data_31fps\_CAIMAN'


#%%
all_tif_names = ot.Get_File_Name(trans_folder,'.mat')

for i,c_name in tqdm(enumerate(all_tif_names)):

    # load current matlab files.
    arrays = {}
    f = scipy.io.loadmat(c_name)
    for k, v in f.items():
        arrays[k] = np.array(v)
    c_stacks = arrays['img_stack'] # the raw u2 tif file.

    # transfer data struct into [2000,512,512]
    c_stacks = np.transpose(c_stacks, (2, 0, 1))

    ## the data have many bad data points, we will do clip if needed. But no clip now.
    c_stacks[:,511,:] = 0
    # save them into subfolders.
    tif.imwrite(ot.join(tif_folder,f'1-001_{i+1}.tif'),c_stacks)

# np.save(day_folder+'\Graph_Stacks.npy',arrays[list(arrays.keys)[0]][:,:,:,0])
# save will cost a fcking year= = transfer data into u2-tif file.
#%% #############################CAIMAN BELOW#############
# run below line by line plz.
Okc = One_Key_Caiman(day_folder, [1],align_base = '1-001',boulder = (20,20,20,20),fps = 31,decay=0.35)
Okc.frame_lists = [29700] # manually add names.
#%%
Okc.Motion_Corr_All()
#%%
#Okc.global_avr = cv2.imread(r'G:\Test_Data\2P\220630_L76_2P\_CAIMAN\Summarize\Global_Average_cai.tif',-1)
Okc.Cell_Find(boulders= Okc.boulder)
# Okc.Series_Generator_Low_Memory()
#%%
Okc.Series_Generator_NG()
#%%######################### CELL CLASS AT LAST ##############



ac = Cell(day_folder,od = False,orien = False,color = False,fps = 31,filter_para = (0.05,5))
ot.Save_Variable(r'D:\#Lee_Data\231201_Lee_Data\231219_Lee_Data_31fps\_CAIMAN','Cell_Class_Raw',ac)

ac.Z_Frames['1-001']

#%% ######################## bin data for video generation #########################
# binned fps are 31/4 ~7Hz.
# only can do this in server.

Yr, dims, T = cm.load_memmap(r'D:\#Lee_Data\231201_Lee_Data\231219_Lee_Data_31fps\_CAIMAN\1-001_0000-#-15-#-1_d1_512_d2_512_d3_1_order_C_frames_29700.mmap')
images = np.reshape(Yr.T, [T] + list(dims), order='F')
# bin and average graph.
images = images[:29696,:,:]
reshaped_frame = images.reshape((3712,8, 512, 512))
# Average the four images along the second axis
averaged_frame = np.mean(reshaped_frame, axis=1)
np.save(ot.join(r'D:\#Lee_Data\231201_Lee_Data\231219_Lee_Data_31fps\_CAIMAN','Raw_Series_bin8.npy'),averaged_frame)

#%% Save this into video.
averaged_frame = np.load(r'D:\#Lee_Data\231201_Lee_Data\231219_Lee_Data_31fps\_CAIMAN\Raw_Series_bin8.npy')
ac = ot.Load_Variable(r'D:\#Lee_Data\231201_Lee_Data\231219_Lee_Data_31fps\_CAIMAN\Cell_Class_Raw.pkl')
#%%
plotable_frames = np.clip(averaged_frame,0,4000)[3000:3500]

frames = (plotable_frames*255/plotable_frames.max())
# frames = plotable_spon*255
#
import cv2
import skvideo.io
import time
width = 512
height = 512
speed = 3 # the post speed.
fps = 31*3/8
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
