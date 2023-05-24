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
from Filters import Signal_Filter

#%%
wp = r'D:\ZR\_Data_Temp\_Temp_Data\220711_temp\UMAP_Datas'
acd = ot.Load_Variable(wp,'All_Series_Dic91.pkl')
sfa = ot.Load_Variable(wp,'L91_Stim_Frame_Align.sfa')# run006 od,007 g16,008 hue7
od_ids = sfa['Run006']
orien_ids = sfa['Run007']
hue_ids = sfa['Run008']
#%%
def Given_Run_Frame(acd,runname = '1-001',fps = 1.301):
    acn = list(acd.keys())
    frame_num = len(acd[1][runname])
    selected_frames = pd.DataFrame(columns = acn,index= list(range(frame_num)))
    for i,cc in enumerate(acn):
        cc_run = acd[cc][runname]
        filted_cc_run = Signal_Filter(cc_run,order =7,filter_para = (0.01/fps,0.6/fps))
        selected_frames[cc] = filted_cc_run
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
for i,cc in enumerate(orien_ids[8]+orien_ids[16]):
    c_frame = orien_frames.loc[cc,:]
    total_label_frame.loc[counter,:] = [c_frame,'G16','8/16','BothEye','Orien157.5','False']
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
for i,cc in enumerate(hue_ids[7]+hue_ids[14]+hue_ids[21]+hue_ids[28]):
    c_frame = hue_frames.loc[cc,:]
    total_label_frame.loc[counter,:] = [c_frame,'Hue7Orien4','7/14/21/28','BothEye','False','White']
    counter +=1    

#%% Calculate numerical ids.
real_total_label_frame = total_label_frame.dropna()
od_label_list = []
orien_label_list = []
hue_label_list = []
for i in range(real_total_label_frame.shape[0]):
    raw_od_label = real_total_label_frame.loc[i,'Eye_Label']
    raw_orien_label = real_total_label_frame.loc[i,'Orien_Label']
    raw_hue_label = real_total_label_frame.loc[i,'Color_Label']
    # label od with number.
    if raw_od_label == 'False':
        od_label_list.append(0)
    elif raw_od_label == 'LE':
        od_label_list.append(1)
    elif raw_od_label == 'RE':
        od_label_list.append(2)
    elif raw_od_label == 'BothEye':
        od_label_list.append(3)
    # label orientation with number.
    if raw_orien_label == 'False':
        orien_label_list.append(0)
    elif raw_orien_label == 'Orien0':
        orien_label_list.append(1)
    elif raw_orien_label == 'Orien22.5':
        orien_label_list.append(2)
    elif raw_orien_label == 'Orien45':
        orien_label_list.append(3)
    elif raw_orien_label == 'Orien67.5':
        orien_label_list.append(4)
    elif raw_orien_label == 'Orien90':
        orien_label_list.append(5)
    elif raw_orien_label == 'Orien112.5':
        orien_label_list.append(6)
    elif raw_orien_label == 'Orien135':
        orien_label_list.append(7)
    elif raw_orien_label == 'Orien157.5':
        orien_label_list.append(8)
    # label color with number.
    if raw_hue_label == 'False':
        hue_label_list.append(0)
    elif raw_hue_label == 'Red':
        hue_label_list.append(1)
    elif raw_hue_label == 'Green':
        hue_label_list.append(2)
    elif raw_hue_label == 'Blue':
        hue_label_list.append(3)
    elif raw_hue_label == 'White':
        hue_label_list.append(4)
# Add three new lines on data frame.
real_total_label_frame['Orien_Label_Num'] = orien_label_list
real_total_label_frame['OD_Label_Num'] = od_label_list
real_total_label_frame['Color_Label_Num'] = hue_label_list
        

#%% save results
ot.Save_Variable(wp,'Frame_ID_infos',real_total_label_frame)
#%% test if avr works.
LE_frames = dict(tuple(real_total_label_frame.groupby('Orien_Label')))['Orien22.5']
avr_graph = np.zeros(651)
for i in range(LE_frames.shape[0]):
    c_frame = np.array(LE_frames.iloc[i,:]['Data'])
    avr_graph += c_frame/LE_frames.shape[0]
#%% plot graphs.
from Cell_Tools.Cell_Visualization import Cell_Weight_Visualization
# recover_graph = np.zeros(shape = (512,512),dtype = 'f8')
# for i in range(651):
#     cc = i+1
#     c_weight = avr_graph[i]
#     cc_x,cc_y = acd[cc]['Cell_Loc']
#     cc_loc = (acd[cc]['Cell_Loc'].astype('i4')[1],acd[cc]['Cell_Loc'].astype('i4')[0])
#     colored_circle_map = cv2.circle(recover_graph,cc_loc,4,c_weight,-1)
recover_graph = Cell_Weight_Visualization(avr_graph,acd)
plt.imshow(recover_graph)