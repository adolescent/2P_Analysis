'''
This script is used for label data 

'''
#%%

import OS_Tools_Kit as ot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Standard_Parameters.Sub_Graph_Dics import Sub_Dic_Generator
import Graph_Operation_Kit as gt
import cv2

#%%
wp = r'D:\ZR\_Data_Temp\_Temp_Data\220711_temp\UMAP_Datas'
acd = ot.Load_Variable(wp,'All_Series_Dic91.pkl')
sfa = ot.Load_Variable(wp,'L91_Stim_Frame_Align.sfa')# run006 od,007 g16,008 hue7
od_ids = sfa['Run006']
orien_ids = sfa['Run007']
hue_ids = sfa['Run008']
#%%
def Given_Run_Frame(acd,runname = '1-001'):
    acn = list(acd.keys())
    frame_num = len(acd[1][runname])
    selected_frames = pd.DataFrame(columns = acn,index= list(range(frame_num)))
    for i,cc in enumerate(acn):
        selected_frames[cc] = acd[cc][runname]
    return selected_frames
OD_frames = Given_Run_Frame(acd,'1-006')
orien_frames = Given_Run_Frame(acd,'1-007')
hue_frames = Given_Run_Frame(acd,'1-008')
total_sample_num = OD_frames.shape[0]+orien_frames.shape[0]+hue_frames.shape[0]
#%% Get dff from F value matrix.
OD_frames = (OD_frames-OD_frames.mean())/OD_frames.mean()
orien_frames = (orien_frames-orien_frames.mean())/orien_frames.mean()
hue_frames = (hue_frames-hue_frames.mean())/hue_frames.mean()
# then calculate z.
OD_frames = OD_frames/OD_frames.std()
orien_frames = orien_frames/orien_frames.std()
hue_frames = hue_frames/hue_frames.std()

#%%
total_label_frame = pd.DataFrame(columns= ['Data','From_Stim','Raw_ID','Eye_Label','Orien_Label','Color_Label'],index = list(range(total_sample_num)))

#%% get labeled data. But first, generate filted dff.
counter = 0
for i,cc in enumerate(od_ids[-1]):# OD ISI
    c_frame = OD_frames.loc[cc,:]
    total_label_frame.loc[counter,:] = [c_frame,'OD','-1','False','False','False']
    counter +=1
for i,cc in enumerate(od_ids[1]):
    c_frame = OD_frames.loc[cc,:]
    total_label_frame.loc[counter,:] = [c_frame,'OD','1','LE','Orien0','False']
    counter +=1
for i,cc in enumerate(od_ids[3]):
    c_frame = OD_frames.loc[cc,:]
    total_label_frame.loc[counter,:] = [c_frame,'OD','3','LE','Orien45','False']
    counter +=1
for i,cc in enumerate(od_ids[5]):
    c_frame = OD_frames.loc[cc,:]
    total_label_frame.loc[counter,:] = [c_frame,'OD','5','LE','Orien90','False']
    counter +=1
for i,cc in enumerate(od_ids[7]):
    c_frame = OD_frames.loc[cc,:]
    total_label_frame.loc[counter,:] = [c_frame,'OD','7','LE','Orien135','False']
    counter +=1
for i,cc in enumerate(od_ids[2]):
    c_frame = OD_frames.loc[cc,:]
    total_label_frame.loc[counter,:] = [c_frame,'OD','2','RE','Orien0','False']
    counter +=1
for i,cc in enumerate(od_ids[4]):
    c_frame = OD_frames.loc[cc,:]
    total_label_frame.loc[counter,:] = [c_frame,'OD','4','RE','Orien45','False']
    counter +=1
for i,cc in enumerate(od_ids[6]):
    c_frame = OD_frames.loc[cc,:]
    total_label_frame.loc[counter,:] = [c_frame,'OD','6','RE','Orien90','False']
    counter +=1
for i,cc in enumerate(od_ids[8]):
    c_frame = OD_frames.loc[cc,:]
    total_label_frame.loc[counter,:] = [c_frame,'OD','8','RE','Orien135','False']
    counter +=1
# Get Orientation tunings.
for i,cc in enumerate(orien_ids[-1]):# G16 ISI
    c_frame = orien_frames.loc[cc,:]
    total_label_frame.loc[counter,:] = [c_frame,'G16','-1','False','False','False']
    counter +=1
for i,cc in enumerate(orien_ids[1]+orien_ids[9]):
    c_frame = orien_frames.loc[cc,:]
    total_label_frame.loc[counter,:] = [c_frame,'G16','1/9','BothEye','Orien0','False']
    counter +=1
for i,cc in enumerate(orien_ids[2]+orien_ids[10]):
    c_frame = orien_frames.loc[cc,:]
    total_label_frame.loc[counter,:] = [c_frame,'G16','2/10','BothEye','Orien22.5','False']
    counter +=1    
for i,cc in enumerate(orien_ids[3]+orien_ids[11]):
    c_frame = orien_frames.loc[cc,:]
    total_label_frame.loc[counter,:] = [c_frame,'G16','3/11','BothEye','Orien45','False']
    counter +=1
for i,cc in enumerate(orien_ids[4]+orien_ids[12]):
    c_frame = orien_frames.loc[cc,:]
    total_label_frame.loc[counter,:] = [c_frame,'G16','4/12','BothEye','Orien67.5','False']
    counter +=1
for i,cc in enumerate(orien_ids[5]+orien_ids[13]):
    c_frame = orien_frames.loc[cc,:]
    total_label_frame.loc[counter,:] = [c_frame,'G16','5/13','BothEye','Orien90','False']
    counter +=1
for i,cc in enumerate(orien_ids[6]+orien_ids[14]):
    c_frame = orien_frames.loc[cc,:]
    total_label_frame.loc[counter,:] = [c_frame,'G16','6/14','BothEye','Orien112.5','False']
    counter +=1
for i,cc in enumerate(orien_ids[7]+orien_ids[15]):
    c_frame = orien_frames.loc[cc,:]
    total_label_frame.loc[counter,:] = [c_frame,'G16','7/15','BothEye','Orien135','False']
    counter +=1    
    
    
# Get color tunings.
for i,cc in enumerate(hue_ids[-1]):
    c_frame = hue_frames.loc[cc,:]
    total_label_frame.loc[counter,:] = [c_frame,'Hue7Orien4','-1','False','False','False']
    counter +=1    
for i,cc in enumerate(hue_ids[1]+hue_ids[8]+hue_ids[15]+hue_ids[22]):
    c_frame = hue_frames.loc[cc,:]
    total_label_frame.loc[counter,:] = [c_frame,'Hue7Orien4','1/8/15/22','BothEye','False','Red']
    counter +=1    
for i,cc in enumerate(hue_ids[3]+hue_ids[10]+hue_ids[17]+hue_ids[24]):
    c_frame = hue_frames.loc[cc,:]
    total_label_frame.loc[counter,:] = [c_frame,'Hue7Orien4','3/10/17/24','BothEye','False','Green']
    counter +=1    
for i,cc in enumerate(hue_ids[5]+hue_ids[12]+hue_ids[19]+hue_ids[26]):
    c_frame = hue_frames.loc[cc,:]
    total_label_frame.loc[counter,:] = [c_frame,'Hue7Orien4','5/12/19/26','BothEye','False','Blue']
    counter +=1    
#%% save frames
real_total_label_frame = total_label_frame.dropna()
ot.Save_Variable(wp,'Frame_ID_infos',real_total_label_frame)
#%% test if avr works.
LE_frames = dict(tuple(real_total_label_frame.groupby('Eye_Label')))['RE']
avr_graph = np.zeros(651)
for i in range(LE_frames.shape[0]):
    c_frame = np.array(LE_frames.iloc[i,:]['Data'])
    avr_graph += c_frame/LE_frames.shape[0]
#%% plot graphs.
recover_graph = np.zeros(shape = (512,512),dtype = 'f8')
for i in range(651):
    cc = i+1
    c_weight = avr_graph[i]
    cc_x,cc_y = acd[cc]['Cell_Loc']
    cc_loc = (acd[cc]['Cell_Loc'].astype('i4')[1],acd[cc]['Cell_Loc'].astype('i4')[0])
    colored_circle_map = cv2.circle(recover_graph,cc_loc,4,c_weight,-1)