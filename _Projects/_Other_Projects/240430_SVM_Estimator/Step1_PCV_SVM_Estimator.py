'''
First, we load in pre-processed data, and generate pca-svm estimation.
The key here is The dim we use for SVM.

'''

#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import SVC
import My_Wheels.OS_Tools_Kit as ot
import seaborn as sns
import copy
from Cell_Class.Advanced_Tools import Z_PCA
from tqdm import tqdm
import pandas as pd



wp = r'D:\#Shu_Data\TailPinch\ChatFlox_Cre\_Stats_All_Points'
all_para = ot.Load_Variable_v2(wp,'All_Raw_Parameters.pkl')
all_para_n = ot.Load_Variable_v2(wp,'Normalized_All_Raw_Parameters.pkl')
all_locs = list(set(all_para['Case']))
savepath = r'D:\#Shu_Data\TailPinch\ChatFlox_Cre\_Stats_All_Points'
All_Paras_Labeled = ot.Load_Variable(savepath,'All_Bio_Paras_Labeled.pkl')
#%% 
'''
Step1, EEG Noise deduction, use advanced method to smoothe EEG plot, remove recording great peak noise, avoiding it's bad response.

'''

#%% EEG Denoise.
# all operation will need to cut the series. so indexing and eeg flat can be done at the same time.
from scipy.signal import find_peaks, medfilt

# trend function. This function will return trend of local data, seems very smooth.
def get_trend(data,  window_size=150):
    """
    Return trend of eeg data, only trend kept.
    """

    # peaks, _ = find_peaks(data, height=peak_height_threshold)
    peaks, _ = find_peaks(data)
    smoothed_data = data.copy()
    for peak in peaks:
        # Calculate the local median around the peak
        start = max(0, peak - window_size // 2)
        end = min(len(data), peak + window_size // 2 + 1)
        local_median = np.median(data[start:end])
        # Replace the peak with the local median
        smoothed_data[peak] = local_median
    # Apply a median filter to the entire time series
    # Ensure that the window size is an odd number
    window_size = window_size if window_size % 2 == 1 else window_size + 1
    smoothed_data = medfilt(smoothed_data, kernel_size=window_size)
    return smoothed_data

# Use trend and local average as level to smooth outliers.
def smooth_extremes(data,trends, window_size=50, threshold =0.1):
    smoothed_data = data.copy()
    # Iterate over the data
    for i in tqdm(range(len(data))):
        # Calculate the local mean and standard deviation
        start = max(0, i - window_size // 2)
        end = min(len(data), i + window_size // 2 + 1)
        local_mean = np.mean(data[start:end])
        local_std = np.std(data[start:end])
        
        # Check if the current value is an extreme value
        if abs(data[i]-trends[i]) > threshold:
            # Replace the extreme value with the local mean
            smoothed_data[i] = trends[i]
    return smoothed_data

# smooth EEG of all data, and put them into a dic.
all_bio_para_dics = {}
plot_flag = True

for i,cloc in enumerate(all_locs):
    cloc_data = all_para_n[all_para_n['Case']==cloc]
    cloc_data = cloc_data.sort_values('Time') 
    raw_eeg = cloc_data['EEG_Below10Hz']

    eeg_trend = get_trend(raw_eeg,200)
    smoothed_eeg = smooth_extremes(raw_eeg,eeg_trend,1000,0.4)

    if plot_flag:
        plt.clf()
        plt.cla()
        plt.plot(raw_eeg)
        plt.plot(smoothed_eeg)
        plt.show()
    cloc_data['EEG_Filted'] = smoothed_eeg
    cloc_data = cloc_data.drop('EEG_Below10Hz',axis = 1)
    all_bio_para_dics[cloc] = cloc_data

#%%
'''
Step2
In the mean time, we will estimate the animal body situation, seperated into 4 parts:
1-AWAKE
2-ANETHESIA
-1 - OTHERS & UNKNOWN, including anes induce and recovery.

It's hard to get very exact estimation, so we only use exacty times, only awake before are added.

'''
## INITIALIZATION, YOU CAN ONLY DO THIS ONCE= =
## JUST BE CAREFUL..
all_bio_para_dics_labeled = {}

#%%
def Stat_Labeler(series_len,last_awake,first_anes,last_anes):
    labels = np.ones(series_len)*-1
    labels[:last_awake] = 1
    labels[first_anes:last_anes] = 2
    return labels
cloc = all_locs[4]
c_para = all_bio_para_dics[cloc]
c_para = c_para.drop('Case',axis = 1)
c_para = c_para.astype('f8')


plt.clf()
plt.cla()
fig,ax = plt.subplots(ncols=1,nrows=3, figsize=(10,8),dpi = 180)
sns.lineplot(data = c_para,x = 'Time',y = 'EEG_Filted',ax = ax[0])
sns.lineplot(data = c_para,x = 'Time',y = 'SpO2',ax = ax[1])
sns.lineplot(data = c_para,x = 'Time',y = 'Pupil',ax = ax[2])
#plt.plot(c_para['Pupil'][:])
#plt.plot(c_para['EEG_Filted'][:])
#%% Do this manually.
# Determine the last awake and first & last anes.
awake_lim = 1500
anes_start = 3000
anes_end = 11000
c_label = Stat_Labeler(len(c_para),awake_lim,anes_start,anes_end)
c_para['State_Label'] = c_label
all_bio_para_dics_labeled[cloc] = c_para

list(all_bio_para_dics_labeled.keys())
#%% Save part, do it after estimation done.
# we need to concat frames first.
for i,cloc in enumerate(all_locs):
    c_frame = all_bio_para_dics_labeled[cloc]
    c_frame['Case'] = cloc
    if i == 0:
        All_Paras_Labeled = c_frame
    else:
        All_Paras_Labeled = pd.concat([All_Paras_Labeled,c_frame])
All_Paras_Labeled = All_Paras_Labeled.reset_index(drop = True)

ot.Save_Variable(savepath,'All_Bio_Paras_Labeled',All_Paras_Labeled)
#%%
'''
Step3, Global PCA Model Establish, we get global PCA to estimate and do dim reduction on all data.
'''

## PCA Model Establishment.

pc_able_data = copy.deepcopy(All_Paras_Labeled)
pc_able_data = pc_able_data.drop('Case',axis = 1)
pc_able_data = pc_able_data.drop('State_Label',axis = 1)
pc_able_data = pc_able_data.drop('Time',axis = 1)
# sns.histplot(data = all_para_n,x = 'EEG_Below10Hz',hue = 'Case')

pcnum = 5
pc_comps,pc_coords,model = Z_PCA(np.array(pc_able_data),sample='Frame',pcnum = pcnum)
# plt.plot(pc_coords[:,1])
plt.scatter(pc_coords[:,0],pc_coords[:,1],s = 1,c = range(pc_coords.shape[0]))

#%%
# Second part will get Awake(1) and Anes(2) Frames to do SVM.
awake_ids = np.where(All_Paras_Labeled['State_Label']==1)[0]
anes_ids = np.where(All_Paras_Labeled['State_Label']==2)[0]

awake_biometers = pc_able_data.iloc[awake_ids,:]
anes_biometers = pc_able_data.iloc[anes_ids,:]

awake_coords = model.transform(np.array(awake_biometers))
anes_coords = model.transform(np.array(anes_biometers))

# and combine awake and stim into 1 single matrix.
all_len = len(awake_ids)+len(anes_ids)
all_labels = np.zeros(all_len)
all_labels[:len(awake_ids)] = 1
all_labels[len(awake_ids):] = 2
all_coords = np.concatenate((awake_coords,anes_coords))


#%%
# Here we do SVM estimation.

svm = SVC(probability=True,kernel='linear')
svm.fit(all_coords, all_labels)
all_label_dists = svm.decision_function(pc_coords)
plt.plot(all_label_dists)

