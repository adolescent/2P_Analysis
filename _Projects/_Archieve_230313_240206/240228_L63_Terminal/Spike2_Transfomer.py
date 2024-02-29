'''
A Little Helper for LC, transfer smr data into mat format.
'''


#%%
import OS_Tools_Kit as ot
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from tqdm import tqdm

wp = r'D:\_Helps\240227_L63_terminal'
all_smr_name = ot.Get_File_Name(wp,'.smr')


#%% Transfer each file.

for i,c_smr_name in tqdm(enumerate(all_smr_name)):
    c_smr_dic = {}
    # c_smr = ot.Spike2_Reader(c_smr_name)
    c_sound_file = np.array(ot.Spike2_Reader(smr_name = c_smr_name,stream_channel = '0')['Channel_Data'])
    c_camera_file = np.array(ot.Spike2_Reader(smr_name = c_smr_name,stream_channel = '1')['Channel_Data'])
    c_smr_dic['Sound_Train'] = c_sound_file
    c_smr_dic['Camera_Train'] = c_camera_file
    savemat(f"{wp}\\{i+1}.mat",c_smr_dic)


