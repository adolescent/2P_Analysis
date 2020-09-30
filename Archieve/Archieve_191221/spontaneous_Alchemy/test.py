# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 15:17:32 2019

@author: ZR
"""

import General_Functions.my_tools as pp
import numpy as np
import matplotlib.pyplot as plt


save_folder = r'E:\ZR\Data_Temp\190412_L74_LM\1-002\results'
spike_data = pp.read_variable(save_folder+r'\\spike_train_Morphology.pkl')
spike_data_filtered = pp.read_variable(save_folder+r'\\spike_train_Morphology_filtered.pkl')
#%%
example_cell = spike_data[135,:]
example_cell_filtered = spike_data_filtered[135,:]
#%%
from scipy import signal

critical_freq = 0.005

order = 10#滤波阶数
capture_rate = 1.301
sos = signal.butter(order,critical_freq,'lowpass',fs = capture_rate,output = 'sos')
filtered = signal.sosfilt(sos, example_cell)
plt.plot(filtered)
#%%
def Test(a,**kwargs):
    test1 = a
    test2 = kwargs
    return test1,test2
a,b = Test(1,d = 2,b = 3,c = 4)
#%%
import random
a = []
b = []
for i in range(250):
    a.append(random.random())
    b.append(random.random())