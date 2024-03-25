'''

Show example of different datas. This will give us an direct image of data we are processing.

'''

#%%
import OS_Tools_Kit as ot
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

wp = r'D:\_Shu_Data\#244_Py_Data'

#%%############################# 1. EEG Reading ############################
eeg_file = np.load(ot.join(wp,'EEG_Python.npy'))

# plot part
# plotable_file = eeg_file[30000:50000,:]
plotable_file = eeg_file[:,:]
fps = 1000
plt.clf()
plt.cla()

# set graph
x_range = np.arange(len(plotable_file))/1000
fig,ax = plt.subplots(nrows=2, ncols=1,figsize = (8,3),dpi = 180,sharex= True)
ax[0].plot(x_range,plotable_file[:,0], color=plt.cm.tab10(0))
ax[1].plot(x_range,plotable_file[:,2], color=plt.cm.tab10(1))
ax[1].set_xlabel('Seconds (s)')

#%% EEG wavelet transformation
import pywt
## Simple version.
# coefficients, frequencies = pywt.cwt(eeg_file[1800000:2700000,2], np.arange(1, 128), 'morl')

# plt.clf()
# plt.cla()
# fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (8,3),dpi = 180,sharex= True)
# sns.heatmap(coefficients, xticklabels=False, yticklabels=False,center = 0,ax = ax,vmax = 2,vmin = -1)
# plt.show()


sampling_rate = 1000  # Hz
desired_frequencies = [30, 10, 5, 0.2]  # Hz
scales = pywt.scale2frequency('morl', desired_frequencies)*sampling_rate
coefficients, frequencies = pywt.cwt(eeg_file[2250000:2750000,2], scales, 'morl')

plt.clf()
plt.cla()
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (8,3),dpi = 180,sharex= True)
sns.heatmap(coefficients, xticklabels=False, yticklabels=False,ax = ax,center = 0)
plt.show()


from Filters import Signal_Filter_v2
# We NEED TO FINISH THIS FUNCTION!

#%% ####################### OTHER EASY DATA ################################# 


import scipy.io

# read in pupil and pad movement
puil_size_data = scipy.io.loadmat(ot.join(wp,'Pupil_Size.mat'))['c_area']
pad_move_data = scipy.io.loadmat(ot.join(wp,'Pad_Speed.mat'))['runspeed']
eeg_data = eeg_file[:,2]

# read temperature data, this is in an csv file.
temperature_data = pd.read_csv(r'D:\_Shu_Data\20240220_#244_chat-flox\BodyTemperature\20240220_#244.csv', skiprows=3,header = None)
temperature_data.set_index(0, inplace=True)
temperature_data = temperature_data.dropna(how='any',axis=1)
temp_freq = len(temperature_data)*10/len(pad_move_data) # from Original Data

# read mouse ox data.
ox_data = pd.read_csv(r'D:\_Shu_Data\20240220_#244_chat-flox\MouseOX\20240220_#244.csv', skiprows=3,header=None)
useful_oc_data = ox_data[[5,6,7]]
useful_oc_data.columns=['SpO2','HR','Resp']

# read in run speed, this time is txt.

runspeed = pd.read_csv(r'D:\_Shu_Data\20240220_#244_chat-flox\RunningSpeed\20240220-speed-.tdms.txt',sep=r'	',skiprows=1,header=None)
runspeed.columns=['Time','X1','Y1','X2','Y2']

#%% Save all paras into 1 single dic.
all_example_dic = {}
all_example_dic['Capture_Frequency'] = pd.DataFrame([1000,10,temp_freq,15],index = ['EEG','Pupil&Pad','Body_Temp','Mouse_Ox'])
all_example_dic['EEG'] = eeg_file[:,2]
all_example_dic['Pupil_Size'] = puil_size_data
all_example_dic['Body_Temp'] = temperature_data
all_example_dic['Mouse_Ox'] = useful_oc_data
all_example_dic['runspeed'] = runspeed
ot.Save_Variable(wp,'All_Example_Infos',all_example_dic)
