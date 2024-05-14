'''
This script will provide advanced data processing method only avaliable on already Z scored data frames.

'''

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.model_selection import cross_val_score
from itertools import groupby
from scipy.fft import fft, ifft
from scipy.fftpack import rfft, irfft
import copy


def Z_PCA(Z_frame,sample = 'Cell',pcnum = 20):
    # INPUT FRAME SHALL BE IN SIZE (N_Frame,N_Cell)
    pca = PCA(n_components = pcnum)
    data = np.array(Z_frame)
    if sample == 'Cell':
        data = data.T# Use cell as sample and frame as feature.
    elif sample == 'Frame':
        data = data
    else:
        raise IOError('Sample method invalid.')
    pca.fit(data)
    PC_Comps = pca.components_# out n_comp*n_feature
    point_coords = pca.transform(data)# in n_sample*n_feature,out n_sample*n_comp
    return PC_Comps,point_coords,pca

def PCNum_Determine(Z_frame,sample = 'Frame',thres = 0.7):
    # if sample == 'Frame':
    pc_num = Z_frame.shape[1]
    # else:
        # pc_num = Z_frame.shape[0]
    _,_,c_model = Z_PCA(Z_frame,sample,pc_num)
    vars = c_model.explained_variance_ratio_
    accomulated_vars = np.cumsum(vars)
    least_loc = np.where(accomulated_vars>thres)[0][0]
    PC_num = least_loc+1
    print(f'We Need {PC_num} PCs to explain {thres} VARs.')
    return PC_num


def Remove_ISI(Z_frame,label):# remove label of raw id -1 and 
    frame_num = label.shape[1]
    non_isi = label.loc['Raw_ID'] != -1
    cutted_label = label.T[non_isi == True]
    non_isi_frame_index = cutted_label.index
    cutted_Z_frame = Z_frame.loc[non_isi_frame_index]
    
    return cutted_Z_frame,cutted_label

def SVM_Classifier(embeddings,label,C = 10):
    classifier = svm.SVC(C = C,probability=True)
    scores = cross_val_score(classifier,embeddings, list(label), cv=5)
    print(f'Score of 5 fold SVC on unsupervised : {scores.mean()*100:.2f}%')
    classifier.fit(embeddings,list(label))
    return classifier,scores.mean()

def SVC_Fit(classifier,data,thres_prob=0.6):
    print('Make sure the classifier is already trained.')
    probas = classifier.predict_proba(data)
    raw_predicted_label = classifier.predict(data)
    predicted_labels = np.zeros(data.shape[0])
    for i in range(probas.shape[0]):
        c_prob = probas[i,:]
        c_max = c_prob.max()
        if c_max<thres_prob:
            predicted_labels[i] =-1
        else:
            # predicted_spon_labels[i] = np.where(c_prob == c_max)[0][0]
            predicted_labels[i] = raw_predicted_label[i]
    refuse_num = np.sum(predicted_labels == -1)
    print(f'{100*refuse_num/data.shape[0]:.2f} % Frames ({refuse_num}/{data.shape[0]}) refused.')
    return predicted_labels

def Average_Each_Label(Z_Frame,Labels):
    label_sets = list(set(Labels))
    acn = list(Z_Frame.columns)
    all_response = pd.DataFrame(0,columns=acn,index = label_sets)
    for i,c_label in enumerate(label_sets):
        c_frame_loc = np.where(Labels == c_label)[0]
        c_frame = Z_Frame.iloc[c_frame_loc,:]
        all_response.loc[c_label,:]=c_frame.mean()
    
    return all_response
############# Tools for series cut and count.
def Label_Event_Cutter(input_series):# input series must be 1/0 frame!
    indices_on = np.where(input_series == True)[0]
    cutted_events = np.split(indices_on, np.where(np.diff(indices_on) != 1)[0]+1)
    all_event_length = np.zeros(len(cutted_events))
    for i,c_series in enumerate(cutted_events):
        all_event_length[i] = len(c_series)
    return cutted_events,all_event_length

# simplyfied version of Label_Event_Cutter, can use directly for interval calculation.
def All_Start_Time(input_series):
    true_indices = np.where(input_series)[0]
    # Find consecutive sequences of True values and their starting indices
    sequences_len = []
    start_index = []
    for k, g in groupby(enumerate(true_indices), lambda ix : ix[0] - ix[1]):
        true_sequence = [x[1] for x in g]
        sequences_len.append(len(true_sequence))
        start_index.append(true_sequence[0])
    # Print the consecutive sequences of True values and their starting indices
    return sequences_len,start_index


def Event_Counter(series): # this function is used to count true list number.
    count = 0
    consecutive_count = 0
    for value in series:
        if value:
            consecutive_count += 1
        else:
            if consecutive_count > 0:
                count += 1
            consecutive_count = 0
    if consecutive_count > 0:
        count += 1
    return count

def Wait_Time_Distribution(start_time):
    wait_times = np.zeros(len(start_time)-1)
    start_time = np.array(start_time)
    for i in range(len(wait_times)):
        c_time = start_time[i]
        if i == 0:
            c_dist = start_time[i+1]-c_time
        else:
            time_before =  start_time[i-1]
            time_after = start_time[i+1]
            c_dist = min(c_time-time_before,time_after-c_time)
        wait_times[i] = c_dist
    return wait_times

################ Tools for shuffle.
def Random_Series_Generator(series_len,event_length):
    combined_series = np.zeros(series_len)
    for j,c_length in enumerate(event_length):
        c_start_loc = np.random.randint(series_len-event_length.max()-2)
        while combined_series[c_start_loc:c_start_loc+int(c_length)+2].sum()!=0:# make no stack.
            c_start_loc = np.random.randint(series_len-event_length.max())

        combined_series[c_start_loc+1:c_start_loc+int(c_length)+1] = 1
    return combined_series

def Spon_Shuffler(spon_frame,method = 'phase',filter_para = (0.005,0.3),fps = 1.301):# 'all' or 'phase'
    shuffled_frame = np.zeros(shape = spon_frame.shape) # output will be an np array, be very careful.
    spon_frame = np.array(spon_frame)
    if method == 'all':
        for i in range(spon_frame.shape[1]):
            # c_series = np.array(spon_frame.iloc[:,i])
            c_series = spon_frame[:,i]
            np.random.shuffle(c_series)
            shuffled_frame[:,i] = c_series
    elif method == 'phase':# do phase shuffle
        # codes below from https://stackoverflow.com/questions/39543002/returning-a-real-valued-phase-scrambled-timeseries
        for i in range(spon_frame.shape[1]):
            # c_series = np.array(spon_frame.iloc[:,i])
            c_series = spon_frame[:,i]
            fs = rfft(c_series)
            power = fs[1:-1:2]**2 + fs[2::2]**2 # strength
            phase = np.arctan2(fs[2::2], fs[1:-1:2]) # in radius
            shuffled_phase = copy.deepcopy(phase)
            start_freq = int(len(power)*filter_para[0]/(fps/2))
            end_freq = int(len(power)*filter_para[1]/(fps/2))
            # shuffle phase of unfilted part only.
            middle_part_phase = copy.deepcopy(phase[start_freq:end_freq])
            np.random.shuffle(middle_part_phase)
            shuffled_phase[start_freq:end_freq] = middle_part_phase
            fsrp = np.sqrt(power[:, np.newaxis]) * np.c_[np.cos(shuffled_phase), np.sin(shuffled_phase)]
            fsrp = np.r_[fs[0], fsrp.ravel(), fs[-1]]
            modified_series = irfft(fsrp)
            shuffled_frame[:,i] = modified_series[:shuffled_frame.shape[0]]# avoid 1 diff problem.
            # fft_result = fft(c_series)
            # magnitude = np.abs(fft_result)
            # phase = np.angle(fft_result)
            # np.random.shuffle(phase)
            # modified_fft_result = magnitude * np.exp(1j * phase)
            # modified_series = ifft(modified_fft_result)
            # # to avoid head-tail bug, we add a 100 len pad for shuffle.
            # # padding_length = 100  # Length of zero-padding
            # # modified_fft_result_padded = np.concatenate((modified_fft_result, np.zeros(padding_length)))
            # # modified_fft_result_padded = np.concatenate((np.zeros(padding_length),modified_fft_result_padded))
            # # modified_series = ifft(modified_fft_result_padded)
            # # shuffled_frame[:,i] = modified_series[padding_length:-padding_length]
            # shuffled_frame[:,i] = modified_series
    elif method == 'dim': # with all cell train unchanged, only change the dim sequence.
        array = np.array(spon_frame).T
        num_rows,_ = array.shape
        indices = np.arange(num_rows)
        np.random.shuffle(indices)
        shuffled_frame = (array[indices, :]).T
    return shuffled_frame

def Shuffle_Multi_Trains(input_series): # the input here must be 0 as null, 1,2,3 as different network types.
    all_stim_types = list(set(input_series))
    all_stim_types.remove(0)
    series_len = len(input_series)
    shuffled_series = np.zeros(series_len)
    for i,c_stim in enumerate(all_stim_types):
        cc_stim_train = input_series == c_stim
        cc_len,_ = All_Start_Time(cc_stim_train)
        cc_len = np.array(cc_len)
        for j,c_length in enumerate(cc_len):
            c_start_loc = np.random.randint(series_len-cc_len.max())
            while shuffled_series[c_start_loc:c_start_loc+int(c_length)].sum()!=0:# make no stack.
                c_start_loc = np.random.randint(series_len-cc_len.max())
            shuffled_series[c_start_loc:c_start_loc+int(c_length)] = c_stim

    return shuffled_series



