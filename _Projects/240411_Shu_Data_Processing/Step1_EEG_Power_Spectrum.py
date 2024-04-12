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


wp = r'D:\_Shu_Data\#244_Py_Data'

eeg_file = np.load(ot.join(wp,'EEG_Python.npy'))[:,2] # only channel 3 is eeg.
all_example_dic = ot.Load_Variablee(wp,'All_Example_Infos.pkl')



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


