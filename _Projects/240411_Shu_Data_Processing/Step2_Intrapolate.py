'''

This function will intrapolate each data into 


'''


#%%
import numpy as np
from scipy.interpolate import interp1d
import OS_Tools_Kit as ot
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pywt
from tqdm import tqdm
from scipy import signal


wp = r'D:\_Shu_Data\#244_Py_Data'

eeg_power = ot.Load_Variable(ot.join(wp,'FFT_Power_fq500Hz_0.5scalar_bin1.pkl')) # only channel 3 is eeg.
all_example_dic = ot.Load_Variable(wp,'All_Example_Infos.pkl')

#%% Basic Parameters.
unify_freq = 1 # Hz
all_sample_1hz = {}
#%% Pad movements & Pupil 
# This is captured in 10Hz, so we just down sample them.

pad_freq = all_example_dic['Capture_Frequency'].loc['Pupil&Pad'][0]
pad_movement = np.linalg.norm(all_example_dic['PadMovement'],axis=1)
decimation_factor = int(pad_freq/unify_freq)
# pad_movement_1Hz = signal.resample(pad_movement, len(pad_movement)//10)
pad_movement_1Hz = np.reshape(pad_movement[:205460],(-1,10)).mean(1)
all_sample_1hz['Pad_Movement'] = pad_movement_1Hz
pupil = all_example_dic['Pupil_Size']
pupil_1Hz = np.reshape(pupil[:205460],(-1,10)).mean(1)
all_sample_1hz['Pupil_Size'] = pupil_1Hz
# plt.plot(pad_movement_1Hz)
#%% Save EEG 
all_sample_1hz['EEG_Spectrum'] = eeg_power['Sepctrum']
all_sample_1hz['theta_delta_ratio'] = eeg_power['Theta']/eeg_power['Delta']
#%% Save Mouse Ox Data.
used_mouse_ox = np.array(all_example_dic['Mouse_Ox'].iloc[:(len(all_example_dic['Mouse_Ox'])//15)*15,:])
used_mouse_ox_1Hz = np.reshape(used_mouse_ox,(len(used_mouse_ox)//15,-1,3)).mean(1)
all_sample_1hz['Oxy_Level'] = used_mouse_ox_1Hz[:,0]
all_sample_1hz['HR'] = used_mouse_ox_1Hz[:,1]
all_sample_1hz['Resp'] = used_mouse_ox_1Hz[:,2]

# plt.plot(used_mouse_ox[:,0])
#%% Save Body Temperature
temp_freq = all_example_dic['Capture_Frequency'].loc['Body_Temp',0]
temp_data = np.array(all_example_dic['Body_Temp'].mean(1))
original_time = np.arange(len(temp_data))/temp_freq
interpolated_func = interp1d(original_time, temp_data, kind='linear')

new_time = np.arange(0, int(original_time[-1])) # up sample to 1 Hz.
interpolated_temperature = interpolated_func(new_time)
all_sample_1hz['Body_Temperature'] = interpolated_temperature

#%% Intrapoalte runspeed and save runspeed.
runspeed_x = np.array(abs(all_example_dic['runspeed']['X1']-all_example_dic['runspeed']['X2']))
runspeed_y = np.array(abs(all_example_dic['runspeed']['Y1']-all_example_dic['runspeed']['Y2']))
original_time = np.array(all_example_dic['runspeed']['Time']/1000)
all_speed = np.sqrt(runspeed_x**2+runspeed_y**2)
interpolated_func = interp1d(original_time,all_speed, kind='linear')
new_time = np.arange(0, int(original_time[-1]))
interpolated_speed = interpolated_func(new_time)
all_sample_1hz['runspeed'] = interpolated_speed

#%% ################ SAVE ALL IN PD FRAMES ####################
min_samplesize = len(all_sample_1hz['theta_delta_ratio']) # EEG is always the least.
all_para_frame = pd.DataFrame(0.0,index = range(min_samplesize),columns = ['theta_delta_ratio','Pupil','PadMovement','BodyTemp','runspeed','SpO2','HR','Resp'])

all_para_frame['theta_delta_ratio'] = all_sample_1hz['theta_delta_ratio'][:min_samplesize]
all_para_frame['Pupil'] = all_sample_1hz['Pupil_Size'][:min_samplesize]
all_para_frame['PadMovement'] = all_sample_1hz['Pad_Movement'][:min_samplesize]
all_para_frame['BodyTemp'] = all_sample_1hz['Body_Temperature'][:min_samplesize]
all_para_frame['runspeed'] = all_sample_1hz['runspeed'][:min_samplesize]
all_para_frame['SpO2'] = all_sample_1hz['Oxy_Level'][:min_samplesize]
all_para_frame['HR'] = all_sample_1hz['HR'][:min_samplesize]
all_para_frame['Resp'] = all_sample_1hz['Resp'][:min_samplesize]

all_sample_1hz['All_Frames'] = all_para_frame

ot.Save_Variable(wp,'All_Samples_1Hz',all_sample_1hz)

#%%
# # Assuming you have your original data points stored in a numpy array called 'data'
# # with shape (300,)
# data = np.arange(300)
# # Calculate the time values corresponding to your original data points
# original_time = np.arange(300) / 0.3

# # Create a function for interpolation
# interpolated_func = interp1d(original_time, data, kind='linear')

# # Create a new time array with the desired frequency (2 Hz) and number of points (2000)
# new_time = np.linspace(0, original_time[-1], num=2000)

# # Interpolate the data at the new time points
# interpolated_data = interpolated_func(new_time)