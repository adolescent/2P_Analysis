# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 12:15:27 2019

@author: ZR
This tool will shuffle spike_train data, including switch cell data in a single frame,
And switch time data in a single cell.
"""

import General_Functions.my_tools as pp
import numpy as np
import random

class Data_Shuffle(object):
    
    name = r'Shuffle datas'
    
    def __init__(self,save_folder,spike_train_name):
        
        self.save_folder = save_folder
        self.spike_train = pp.read_variable(save_folder+r'\\'+spike_train_name)
        self.N_cell,self.N_frame = np.shape(self.spike_train)
        
        
    def shuffle_in_frame(self):#This part will shuffle cell in every frame, with frame series not change
        
        shuffled_data_in_frame = np.zeros(shape = (self.N_cell,self.N_frame),dtype = np.float64)
        for i in range(self.N_frame):
            shuffled_data_in_frame[:,i] = random.sample(list(self.spike_train[:,i]),self.N_cell)
        pp.save_variable(shuffled_data_in_frame,save_folder+r'\\spike_train_shuffle_in_frame.pkl')
        
    def shuffle_cross_frames(self):#This part will shuffle every spike train across frame, with cell location not change
        
        shuffled_data_cross_frames = np.zeros(shape = (self.N_cell,self.N_frame),dtype = np.float64)
        for i in range(self.N_cell):
            shuffled_data_cross_frames[i,:] = random.sample(list(self.spike_train[i,:]),self.N_frame)
        pp.save_variable(shuffled_data_cross_frames,save_folder+r'\\spike_train_shuffle_cross_frames.pkl')
    
    
if __name__ == '__main__':
    save_folder = r'E:\ZR\Data_Temp\190412_L74_LM\1-002\results'
    spike_train_name = 'spike_train_Morphology_filtered.pkl'
    DS = Data_Shuffle(save_folder,spike_train_name)
    DS.shuffle_in_frame()
    DS.shuffle_cross_frames()

# =============================================================================
#     spike_train = DS.spike_train
#     a = pp.read_variable(save_folder+r'\\spike_train_shuffle_in_frame.pkl')
#     b = pp.read_variable(save_folder+r'\\spike_train_shuffle_cross_frames.pkl')
# =============================================================================
