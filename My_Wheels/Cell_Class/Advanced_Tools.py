'''
This script will provide advanced data processing method only avaliable on already Z scored data frames.

'''

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.model_selection import cross_val_score


def Z_PCA(Z_frame,sample = 'Cell'):
    pca = PCA()
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
    print(f'Score of 5 fold SVC on OD unsupervised : {scores.mean()*100:.2f}%')
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

