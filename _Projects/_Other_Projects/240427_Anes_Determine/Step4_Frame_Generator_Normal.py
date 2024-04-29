

#%%

import OS_Tools_Kit as ot
import neo
import numpy as np
from neo.io import PlexonIO
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd

all_folders = ot.Get_Subfolders(r'D:\#Shu_Data\Data_SpontaneousWakeUp','#')


#%%
save_path = r'D:\#Shu_Data\Data_SpontaneousWakeUp\_Stats_All_Points'

for i,c_loc in enumerate(all_folders):
    all_para_dic = ot.Load_Variable(ot.join(c_loc,'_Processing'),'All_Bio_Paras_1Hz.pkl')
    series_len = min(len(all_para_dic['Ox']),len(all_para_dic['Pupil']),len(all_para_dic['Pad_Movement']))
    bio_para_frame = pd.DataFrame(0.0,index = range(series_len),columns = ['SpO2','HR','Resp','Pluse_Distention','Pad_Movement','Pupil'])
    bio_para_frame[['SpO2','HR','Resp','Pluse_Distention']] = all_para_dic['Ox'].iloc[:series_len]
    bio_para_frame['Pad_Movement'] = all_para_dic['Pad_Movement'][:series_len]
    bio_para_frame['Pupil'] = all_para_dic['Pupil'][:series_len]
    ot.Save_Variable(ot.join(c_loc,'_Processing'),'Para_Frames',bio_para_frame)


#%%
save_path = r'D:\#Shu_Data\Data_SpontaneousWakeUp\_Stats_All_Points'
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
