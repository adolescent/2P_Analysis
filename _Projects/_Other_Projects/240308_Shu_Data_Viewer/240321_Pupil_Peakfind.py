



#%%
import OS_Tools_Kit as ot
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy
from My_Wheels.Filters import Signal_Filter_v2
import h5py


wp = r'D:\_Shu_Data\240321_Pupil'
pupil_file = h5py.File(ot.join(wp,'20240315_#1006VideoTrim_Frame1-216000_proc.mat'))
# pupil_file = h5py.File(ot.join(wp,'#1002_20240309_proc.mat'))
# pupil_file = h5py.File(ot.join(wp,'#1004_20240312_proc.mat'))
pupil_data = np.array(pupil_file['proc']['pupil']['area']).flatten()
fps = 10

#%% Step1, file pupil data to get flat and pepper less signal.
used_pupil_data = pupil_data[10:-10]

# Keep power between 0.005-0.5Hz
filted_pupil_data = Signal_Filter_v2(series=used_pupil_data,HP_freq=0.01,LP_freq=0.5,fps = fps,keep_DC=False)

# plt.plot(filted_pupil_data[50000:100000])
plt.clf()
plt.cla()
fig,ax = plt.subplots(ncols=1,nrows=2,dpi = 180,sharex=True,figsize = (8,3))
ax[0].plot(used_pupil_data, color=plt.cm.tab10(0),label = 'Raw Data')
ax[1].plot(filted_pupil_data, color=plt.cm.tab10(1),label = 'Filted Data')
ax[0].legend()
ax[1].legend()


#%% Step2 Set window size,step.

win_len =  600 # in seconds
win_step = 120

win_len_frame = win_len*fps
win_step_frame = win_step*fps

def cut_series(series, width, step):
    num_windows = (len(series) - width) // step + 1
    windows = np.zeros((num_windows, width))
    for i in range(num_windows):
        start = i * step
        end = start + width
        windows[i] = series[start:end]
    return windows

cutted_windows = cut_series(filted_pupil_data,win_len_frame,win_step_frame)
# cutted_windows = cut_series(used_pupil_data,win_len_frame,win_step_frame)

#%% Step3, Find peaks in each window.
from scipy.signal import find_peaks
peaks_thres = 2 # in std
peak_width = 5 # in second
peak_dist = 5 # in second
absolute_lim = 0
all_peak_locs = []
all_info = []
win_num = len(cutted_windows)
all_win_peak_info = pd.DataFrame(0.0,index = range(win_num),columns = ['Time','Peak_Num','Peak_Height'])
for i in range(win_num):
    # get curernt time window
    c_window = cutted_windows[i,:]
    # find peaks of current window
    c_peak_thres = c_window.mean()+c_window.std()*peaks_thres
    final_peak_thres = max(c_peak_thres,absolute_lim)
    # final_peak_thres = absolute_lim+c_window.mean()
    peaks, info = find_peaks(c_window, height=final_peak_thres,distance=peak_dist*fps,width = peak_width*fps,prominence=final_peak_thres)
    all_info.extend(info['peak_heights']/info['widths'])
    # get peak location and peak heights
    c_peak_heights = c_window[peaks].mean()
    peaks_loc = np.array(peaks)+win_step_frame*i
    all_peak_locs.extend(list(peaks_loc))
    all_win_peak_info.iloc[i,:] = [i*win_step+win_len/2,len(peaks_loc),c_peak_heights]

real_peaks = list(set(all_peak_locs))
real_peaks.sort()
all_win_peak_info = all_win_peak_info.fillna(0)

# plot peaks in data.
plt.clf()
plt.cla()
fig,ax = plt.subplots(ncols=1,nrows=2,dpi = 180,sharex=True,figsize = (8,4))


# Plot real data on graph
x = used_pupil_data
ax[0].plot(x,color=plt.cm.tab10(0),label = 'Raw Data')
ax[0].plot(real_peaks, x[real_peaks], "x",color = 'b')
# ax[0].plot(np.zeros_like(x), "--", color="gray")
x2 = filted_pupil_data
ax[1].plot(x2, color=plt.cm.tab10(1),label = 'Filted Data')
ax[1].plot(real_peaks, x2[real_peaks], "x",color = 'b')
ax[0].legend()
ax[1].legend()
ax[0].set_title('Peaks On Real Data')
ax[1].set_title('Peaks On Filted Data')
fig.tight_layout()

#%% And Plot Peak Analysis by Time Here.

plt.clf()
plt.cla()
fig,ax = plt.subplots(ncols=1,nrows=2,dpi = 180,sharex=True,figsize = (8,4))
ax[0].plot(used_pupil_data[::10]/2, color=plt.cm.tab10(0),label = 'Filted Data',alpha = 0.2)
ax[1].plot(used_pupil_data[::10]/30, color=plt.cm.tab10(0),label = 'Filted Data',alpha = 0.2)
sns.lineplot(data = all_win_peak_info,x = 'Time',y = 'Peak_Height',ax = ax[0],color=plt.cm.tab10(1))
sns.lineplot(data = all_win_peak_info,x = 'Time',y = 'Peak_Num',ax = ax[1],color=plt.cm.tab10(1))

ax[0].set_ylim(-10,all_win_peak_info['Peak_Height'].max()+20)
ax[1].set_ylim(0,all_win_peak_info['Peak_Num'].max()+5)
ax[0].set_title('All Peak Heights Avr',size = 12)
ax[1].set_title('Peak Number Counts',size = 12)
fig.tight_layout()

#%% Maybe a phase lag between height and counts?
plt.clf()
plt.cla()
fig,ax = plt.subplots(ncols=1,nrows=1,dpi = 180,sharex=True,figsize = (8,4))
# plotabel_data = pd.melt(frame=all_win_peak_info,id_vars=['Time'],value_vars = ['Peak_Num','Peak_Height'])
# sns.lineplot(data = plotabel_data,x = 'Time',y = 'value',ax = ax,hue = 'variable',size_norm=True)

sns.lineplot(data = all_win_peak_info,x = 'Time',y = 'Peak_Num',ax = ax,color=plt.cm.tab10(0),label = 'Peak Num')
ax2 = ax.twinx()
sns.lineplot(data = all_win_peak_info,x = 'Time',y = 'Peak_Height',ax = ax2,color=plt.cm.tab10(1),label = 'Peak Height')
