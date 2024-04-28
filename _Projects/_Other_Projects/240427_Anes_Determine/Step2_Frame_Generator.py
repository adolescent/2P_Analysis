''''
After preprocessing, we can generate single data frame of all data points.

'''


#%%
import OS_Tools_Kit as ot
import neo
import numpy as np
from neo.io import PlexonIO
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd

all_folders = ot.Get_Subfolders(r'D:\#Shu_Data\TailPinch\ChatFlox_Cre','#')

#%% # 243 Processing.
c_path = all_folders[0]
all_para_dic = ot.Load_Variable(ot.join(c_path,'_Processing'),'All_Bio_Paras_1Hz.pkl')
# all_paras = list(all_para_dic.keys())
# plt.plot(all_para_dic['EEG']['Sepctrum'][:20].mean(0))
# Just Assume the last time of Ox as the last time of pad.
pad_len = len(all_para_dic['Pad_Movement'])
ox_len = len(all_para_dic['Ox'])
ox_start_time = pad_len-ox_len
# cut all series to ox len, EEG as the least.
eeg_10hz = all_para_dic['EEG']['Sepctrum'][:20].mean(0)[ox_start_time:]
series_len = len(eeg_10hz)
bio_para_frame = pd.DataFrame(0.0,index = range(series_len),columns = ['EEG_Below10Hz','SpO2','HR','Resp','Pluse_Distention','Pad_Movement','Pupil','RunSpeed','BodyTemp'])
bio_para_frame['EEG_Below10Hz'] = eeg_10hz
# this case ox not start from 0.
bio_para_frame[['SpO2','HR','Resp','Pluse_Distention']] = all_para_dic['Ox'].iloc[:series_len]

bio_para_frame['Pad_Movement'] = all_para_dic['Pad_Movement'][ox_start_time:ox_start_time+series_len]
bio_para_frame['Pupil'] = all_para_dic['Pupil'][ox_start_time:ox_start_time+series_len]
bio_para_frame['RunSpeed'] = all_para_dic['RunSpeed'][ox_start_time:ox_start_time+series_len]
bio_para_frame['BodyTemp'] = all_para_dic['Bode_Temp'][ox_start_time:ox_start_time+series_len]
ot.Save_Variable(ot.join(c_path,'_Processing'),'Para_Frames',bio_para_frame)
#%% For other runs, just use standard method.
for i,c_path in enumerate(all_folders[1:]):
    all_para_dic = ot.Load_Variable(ot.join(c_path,'_Processing'),'All_Bio_Paras_1Hz.pkl')
    eeg_10hz = all_para_dic['EEG']['Sepctrum'][:20].mean(0)
    series_len = len(eeg_10hz)
    bio_para_frame = pd.DataFrame(0.0,index = range(series_len),columns = ['EEG_Below10Hz','SpO2','HR','Resp','Pluse_Distention','Pad_Movement','Pupil','RunSpeed','BodyTemp'])
    bio_para_frame['EEG_Below10Hz'] = eeg_10hz
    bio_para_frame[['SpO2','HR','Resp','Pluse_Distention']] = all_para_dic['Ox'].iloc[:series_len]
    bio_para_frame['Pad_Movement'] = all_para_dic['Pad_Movement'][:series_len]
    bio_para_frame['Pupil'] = all_para_dic['Pupil'][:series_len]
    bio_para_frame['RunSpeed'] = all_para_dic['RunSpeed'][:series_len]
    bio_para_frame['BodyTemp'] = all_para_dic['Bode_Temp'][:series_len]
    ot.Save_Variable(ot.join(c_path,'_Processing'),'Para_Frames',bio_para_frame)

#%% Generate an overall frame
save_path = r'D:\#Shu_Data\TailPinch\ChatFlox_Cre\_Stats_All_Points'
for i,c_loc in enumerate(all_folders):
    c_case = c_loc.split('\\')[-1]
    c_frame = ot.Load_Variable(ot.join(c_loc,'_Processing'),'Para_Frames.pkl')
    c_frame['Time'] = list(range(len(c_frame)))
    c_frame['Case'] = c_case
    # and we add Z score data here,
    z_c_frame = (c_frame-c_frame.mean(0))/c_frame.std(0)
    z_c_frame['Time'] = list(range(len(c_frame)))
    z_c_frame['Case'] = c_case
    if i == 0:
        all_frame = c_frame
        all_frame_z = z_c_frame
    else:
        all_frame = pd.concat((all_frame,c_frame))
        all_frame_z = pd.concat((all_frame_z,z_c_frame))
ot.Save_Variable(save_path,'All_Raw_Parameters',all_frame)
ot.Save_Variable(save_path,'Normalized_All_Raw_Parameters',all_frame_z)