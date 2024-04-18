'''
We keep the possibility of wavelet usage, but we use slide window FFT here.

'''


#%%
import OS_Tools_Kit as ot
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pywt
from tqdm import tqdm


wp = r'D:\_Shu_Data\#244_Py_Data'

eeg_file = np.load(ot.join(wp,'EEG_Python.npy'))[:,2] # only channel 3 is eeg.
all_example_dic = ot.Load_Variable(wp,'All_Example_Infos.pkl')



#%% ##########################WAVELET METHOD, DECREPTED######################
## this is very costy and memory eating.
# binsize = 10
# used_eeg = eeg_file[:,2].reshape(-1,binsize).mean(1)

# all_freq = np.arange(0.5,30.5,0.5)
# sampling_rate = 1000/binsize  # Hz
# scales = pywt.scale2frequency('morl', all_freq)*sampling_rate
# coefficients, frequencies = pywt.cwt(used_eeg, scales, 'morl')
# delta_band = (0.5,4)
# theta_band = (4,7)
# alpha_band = (8,13)
# beta_band = (13,30)
# def Get_Band_Power(coefficients,all_freq,used_band):
#     start_line = np.where(all_freq>=used_band[0])[0][0]
#     end_line = np.where(all_freq<=used_band[1])[0][-1]
#     band_power = coefficients[start_line:end_line,:].mean(0)
#     return band_power

# delta_power = Get_Band_Power(coefficients,all_freq,delta_band)
# theta_power = Get_Band_Power(coefficients,all_freq,theta_band)

#%% ############################# SLIDE WINDOW FFT METHOD. ##########################
binsize = 1
real_fps = 1000/binsize

winsize = 60 # seconds
step =  1# seconds
win_pointnum = winsize*real_fps
step_pointnum = step*real_fps

# bin eeg first.
bin_num = len(eeg_file)//binsize
used_eeg = eeg_file[:bin_num*binsize].reshape(bin_num,binsize).mean(1)


def Transfer_Into_Freq(input_matrix,freq_bin = 0.5,fps = 1.301):
    input_matrix = np.array(input_matrix)
    # get raw frame spectrums.
    all_specs = np.zeros(shape = ((input_matrix.shape[0]// 2)-1,input_matrix.shape[1]),dtype = 'f8')
    for i in range(input_matrix.shape[1]):
        c_series = input_matrix[:,i]
        c_fft = np.fft.fft(c_series)
        power_spectrum = np.abs(c_fft)[1:input_matrix.shape[0]// 2] ** 2
        power_spectrum = power_spectrum/power_spectrum.sum()
        all_specs[:,i] = power_spectrum
    
    binnum = int(fps/(2*freq_bin))
    binsize = round(len(all_specs)/binnum)
    binned_freq = np.zeros(shape = (binnum,input_matrix.shape[1]),dtype='f8')
    for i in range(binnum):
        c_bin_freqs = all_specs[i*binsize:(i+1)*binsize,:].sum(0)
        binned_freq[i,:] = c_bin_freqs
    return binned_freq

# get slide window fft.
winnum = int(((len(used_eeg)-win_pointnum)//step_pointnum)+1)
freq_bin = 0.5
slided_ffts = np.zeros(shape = (int(real_fps/(2*freq_bin)),winnum),dtype = 'f8')

for i in tqdm(range(winnum)):
    c_win = used_eeg[int(i*step_pointnum):int(i*step_pointnum+win_pointnum)]
    slided_ffts[:,i] = Transfer_Into_Freq(c_win.reshape(-1,1),freq_bin,real_fps).flatten()

# get each band's power.
delta_power = slided_ffts[1:8,:].sum(0)
theta_power = slided_ffts[8:14,:].sum(0)
alpha_power = slided_ffts[16:26,:].sum(0)
beta_power = slided_ffts[26:60,:].sum(0)
all_power = {}
all_power['Sepctrum'] = slided_ffts
all_power['Delta'] = delta_power # 0.5-4Hz
all_power['Theta'] = theta_power # 4-7Hz
all_power['Alpha'] = alpha_power # 8-13Hz
all_power['Beta'] = beta_power # 13-30Hz

ot.Save_Variable(wp,'FFT_Power_fq500Hz_0.5scalar_bin1',all_power)
#%% Plot all band power spectrum and each band's power.
plt.cla()
plt.clf()

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(40,4),dpi = 180)
sns.heatmap(slided_ffts[:60,:],center = 0,vmax=0.15,ax = ax,xticklabels=False,yticklabels=False,)
ax.invert_yaxis()

ax.set_title('FFT Power Spectrum')
# ax.set_yticks(np.linspace(0,60,11))
# ax.set_yticklabels(np.linspace(0,60,11)*0.5)
ax.set_yticks(np.linspace(0,60,7))
ax.set_yticklabels(np.linspace(0,60,7)*0.5)
ax.set_ylabel('Freq. Power Propotion')


ax.set_xticks(np.linspace(0,330,12)*60/step)
ax.set_xticklabels(np.linspace(0,330,12))
ax.set_xlabel('Time(s)')

#%% plot each band power.
plt.cla()
plt.clf()
fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(30,15),dpi = 180,sharex= True)
sns.heatmap(slided_ffts[:60,:],center = 0,vmax=0.15,ax = axes[0],xticklabels=False,yticklabels=False,cbar=False)
axes[1].plot(theta_power/delta_power,color=plt.cm.tab10(0))
axes[2].plot(delta_power,color=plt.cm.tab10(1))
axes[3].plot(theta_power,color=plt.cm.tab10(2))
axes[4].plot(alpha_power,color=plt.cm.tab10(3))
axes[5].plot(beta_power,color=plt.cm.tab10(4))


axes[5].set_xticks(np.linspace(0,330,12)*60/step)
axes[5].set_xticklabels(np.linspace(0,330,12))
axes[5].set_xlabel('Time(s)')

axes[0].set_yticks(np.linspace(0,60,7))
axes[0].set_yticklabels(np.linspace(0,60,7)*0.5)
axes[0].set_ylabel('Freq. Power Propotion')
axes[1].set_ylabel('Theta/Delta Ratio')
axes[2].set_ylabel('Delta Band')
axes[3].set_ylabel('Delta Band')
axes[4].set_ylabel('Alpha Band')
axes[5].set_ylabel('Beta Band')


