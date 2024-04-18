'''
Try to read in and filt my ECG signals.
'''
#%%
import OS_Tools_Kit as ot
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Filters import Signal_Filter_v2,Signal_Filter
import neo

file_path = r'D:\#FDU\240314_EEG_Test\20240313-te4-M1.smr'
signals = ot.Spike2_Reader(smr_name=file_path,stream_channel='0')
#%%
binsize = 40
ecg_data = np.array(signals['Channel_Data']).flatten()
num_groups = len(ecg_data) // binsize
# Reshape the data into complete 10-frame groups
ecg_down = ecg_data[:num_groups * binsize].reshape(-1, binsize)
ecg_down = np.mean(ecg_down, axis=1)
# ecg_down = ecg_down[:,0]
#%%
# Compute the average of each group

used_time = [230,250]
fps = 10000/binsize
# a = Signal_Filter(ecg_down,order = 5,filter_para = (40*2/fps,False))
# a = Signal_Filter(ecg_down,order = 3,filter_para = (False,20*2/fps),dc_keep=False)
# a = Signal_Filter(a,order = 3,filter_para = (0.5*2/fps,False),dc_keep=False)
a = Signal_Filter_v2(ecg_down,order = 5,HP_freq=1,LP_freq=30,fps=fps,keep_DC=False)
used_frame = a[int(used_time[0]*fps):int(used_time[1]*fps)]


plt.clf()
plt.cla()
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,3),dpi = 180)
ax.plot(np.array(range(len(used_frame)))/fps,used_frame*1000)
ax.set_title('ECG Signals')
ax.set_ylabel('Voltage (mV)')
ax.set_xlabel('Time (s)')


#%% calculate fft power
from Analyzer.My_FFT import FFT_Power
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9,4),dpi = 180)
ax.plot(FFT_Power(a,fps = fps)[:5])
# ax.set_ylim(0,10)
