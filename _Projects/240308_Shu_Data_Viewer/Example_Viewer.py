'''

Show example of different datas. This will give us an direct image of data we are processing.

'''

#%%
import OS_Tools_Kit as ot
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

wp = r'D:\_Shu_Data\20240220_#244_chat-flox\EEG'

#%%############################# 1. EEG Reading
eeg_file = np.load(ot.join(wp,'EEG_Python.npy'))

#%% plot part
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

#%% wavelet transformation
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
