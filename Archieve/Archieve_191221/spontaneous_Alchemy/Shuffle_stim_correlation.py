# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 14:23:26 2019

@author: ZR
This function will calculate correlation between stimulus graph & all shuffle series
In return, a dictionary will record average and std of every frame.
"""

import General_Functions.my_tools as pp
import numpy as np
import scipy.stats


class Shuffle_Correlation(object):
    
    name = r'Shuffle correlation plots'
    
    def __init__(self,save_folder,graph_folder):
    
        self.save_folder = save_folder
        self.graph_folder = graph_folder
        
    def read_in(self):
    
        #Read in all stim graph first
        all_graph_name = pp.file_name(self.graph_folder,'.pkl')
        self.stim_graph_set = {}
        for i in range(len(all_graph_name)):
            name = all_graph_name[i].split('\\')[-1][0:-10]
            self.stim_graph_set[name] = pp.read_variable(all_graph_name[i])
        
        ##Then read in all shuffled trains as a vector.
        shuffled_folder = save_folder+r'\Shuffled_trains'
        all_shuffled_train_name = pp.file_name(shuffled_folder,'.pkl')
        self.shuffle_times = len(all_shuffled_train_name)
        self.Cell_Num,self.Frame_Num = np.shape(pp.read_variable(all_shuffled_train_name[0]))#Use train0 to read shapes
        self.shuffle_matrix = np.zeros(shape = (self.shuffle_times,self.Cell_Num,self.Frame_Num),dtype = np.float64)
        for i in range(self.shuffle_times):
            self.shuffle_matrix[i,:,:] = pp.read_variable(all_shuffled_train_name[i])
        
    def single_frame_calculation(self,stim_graph,frame_Num):#give in stim graph and frame, return std and mean.
        
        self.correlation_distribution = []# get the distribution of Pearson R
        for i in range(self.shuffle_times):
            r,_ = scipy.stats.pearsonr(self.shuffle_matrix[i,:,frame_Num],stim_graph[:,0])
            self.correlation_distribution.append(r)
        mean = np.mean(self.correlation_distribution)
        std = np.std(self.correlation_distribution)
        return mean,std
    
    def main_calculation(self):
        
        self.Shuffle_Dictionary = {}
        for i in range(len(self.stim_graph_set)):#Cycle all stim graphs
            current_name = list(self.stim_graph_set.keys())[i]
            current_stim_graph = list(self.stim_graph_set.values())[i]
            current_correlation = np.zeros(shape = [3,self.Frame_Num],dtype = np.float64)#3D series, mean+-2.5std
            for j in range(self.Frame_Num):#Then cycle all frames
                mean,std = self.single_frame_calculation(current_stim_graph,j)
                current_correlation[0,j] = mean-2.5*std
                current_correlation[1,j] = mean
                current_correlation[2,j] = mean+2.5*std
            self.Shuffle_Dictionary[current_name] = current_correlation
        pp.save_variable(self.Shuffle_Dictionary,save_folder+r'\\Stim_Shuffle_Correlation.pkl')
    
if __name__ == '__main__':
    
    graph_folder = r'E:\ZR\Data_Temp\190412_L74_LM\All-Stim-Maps\Run02'
    save_folder = r'E:\ZR\Data_Temp\190412_L74_LM\1-002\results'
    SC = Shuffle_Correlation(save_folder,graph_folder)
    SC.read_in()
    SC.main_calculation()
    test_1 = SC.stim_graph_set
    test_2 = SC.Shuffle_Dictionary