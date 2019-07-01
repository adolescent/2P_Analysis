# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 14:59:29 2019

@author: ZR
"""

#%% Test filter

import General_Functions.my_tools as pp
import matplotlib.pyplot as plt
import numpy as np 
from scipy import signal
from scipy.stats import pearsonr




spike_train = pp.read_variable(r'E:\ZR\Data_Temp\190412_L74_LM\1-001\results\spike_train.pkl')
plot_sample = spike_train[53,:]
#plt.plot(plot_sample)
freq = np.fft.fft(plot_sample)
sos = signal.butter(10,0.0000001,'highpass',fs = 1.301,output = 'sos')
#建立滤波器的二项式表达，顺序是阶数,[低高频]，类型，采样频率，输出sos
filtered = signal.sosfilt(sos, plot_sample)
freq_filtered = abs(np.fft.fft(filtered))
plt.plot(filtered[0:200])
pearsonr(plot_sample,filtered)[0]

#%% Test band cut
freq_cut = freq
freq_cut[0:100]=0
freq_cut[1481:1581] = 0
test = np.fft.ifft(freq_cut)
#plt.plot(test)
plt.plot(abs(np.fft.fft(test)))