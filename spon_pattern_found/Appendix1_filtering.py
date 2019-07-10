# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 14:49:15 2019

@author: ZR

低通滤波器可能会导致相位出现一个延迟，延迟的时间和低通的频率有关。
因此，在处理时我们选择高通滤波。
"""

import General_Functions.my_tools as pp
from scipy import signal
import numpy as np 

#%% 这里是参数设定，决定了滤波的个性化设计
critical_freq = 0.01#截止频率，即这个频率以上的信号可以通过
save_folder = r'E:\ZR\Data_Temp\190412_L74_LM\1-001\results'

#%%这里是默认参数，一般不需要修改，不过仍要注意
order = 10#滤波阶数
capture_rate = 1.301#这个是采样频率，对GA就是1.301，RG要看bin
spike_train = pp.read_variable(save_folder+r'\\spike_train_Morphology.pkl')

#%%滤波核心部分
spike_train_filtered = np.zeros(shape = np.shape(spike_train),dtype = np.float64)
for i in range(np.shape(spike_train)[0]):
    sos = signal.butter(order,critical_freq,'highpass',fs = capture_rate,output = 'sos')
    filtered_temp = signal.sosfilt(sos, spike_train[i,:])
    spike_train_filtered[i,:] = filtered_temp
    
pp.save_variable(spike_train_filtered,save_folder+r'\\spike_train_Morphology_filtered.pkl')