# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 14:38:01 2019

@author: ZR
This function will generate the correlation plot between stimulus map and spontaneous series,
in order to understand pattern repeats.
"""
import numpy as np
import General_Functions.my_tools as pp
import scipy.stats
import matplotlib.pyplot as plt


class Correlation_Functions(object):
    
    name = r'Correlation between stim map and spon series'
    
    def __init__(self,save_folder,graph_folder,spike_train,shuffle_flag):
        
        self.save_folder = save_folder
        self.graph_folder = graph_folder
        self.spike_train = spike_train#The spike train of spontaneous dF/F
        self.N_Cell,self.N_frame = np.shape(self.spike_train)
        self.shuffle_flag = shuffle_flag
        if shuffle_flag == True:
            self.Shuffled_Dictionary = pp.read_variable(save_folder+r'\\Stim_Shuffle_Correlation.pkl')
        
        
    def stim_graph_cycle(self):#Read in all
        
        self.all_stim_graph = {}
        all_graph_name = pp.file_name(self.graph_folder,'.pkl') 
        for i in range(len(all_graph_name)):
            temp_graph = pp.read_variable(all_graph_name[i])[:,0]
            temp_name = all_graph_name[i].split('\\')[-1][:-10]
            self.all_stim_graph[temp_name] = temp_graph
    
    def correlation_calculator(self,stim_graph):#This function calculate 1 graph
        
        correlation_plot = np.zeros(shape = (self.N_frame,1),dtype = np.float64)
        for i in range(self.N_frame):
            correlation_plot[i],_ = scipy.stats.pearsonr(self.spike_train[:,i],stim_graph)
        return correlation_plot#This function will return a correlation plot.   


     
    
    def plot_correlation_single(self):
        
        all_keys = list(self.all_stim_graph.keys())
        for i in range(len(all_keys)):#cycle all stim graphs
            plt.figure(figsize = (200,7))
            
            if self.shuffle_flag ==True:# if true, add shuffle data in it.
                down_lim = self.Shuffled_Dictionary[all_keys[i]][0]
                shuffle_avr = self.Shuffled_Dictionary[all_keys[i]][1]
                up_lim = self.Shuffled_Dictionary[all_keys[i]][2]
            plt.fill_between(range(self.N_frame),up_lim,0,color = '#dfdfdf')
            plt.fill_between(range(self.N_frame),down_lim,0,color = '#dfdfdf')
            plt.plot(shuffle_avr,color = '#8d8d8d')
            
            current_plot = self.correlation_calculator(self.all_stim_graph[all_keys[i]])
            plt.plot(current_plot,label=all_keys[i])
            plt.legend()
            plt.title('Correlation Plot of Spon & '+all_keys[i]+' Graph')
            plt.xlabel('Frames')
            plt.ylabel('Pearson R')
            plt.grid()#网格
            plt.axhline(y = 0,color = '#d62728')#红色
            plt.ylim((-0.4,0.4))
            plt.xlim((0,self.N_frame))
            x_ticks = np.arange(0,self.N_frame,50)
            plt.xticks(x_ticks)
            correlation_folder = save_folder+r'\\Correlation_vs_Stim_Maps'
            pp.mkdir(correlation_folder)
            plt.savefig(correlation_folder+r'\\'+all_keys[i]+'.png')
            plt.close('all')
            
    def plot_all(self):
        
        all_keys = list(self.all_stim_graph.keys())
        plt.figure(figsize = (200,7))
        plt.title('Correlation Plot of Spon & All plots Graph')
        plt.xlabel('Frames')
        plt.ylabel('Pearson R')
        plt.grid()#网格
        plt.axhline(y = 0,color = '#d62728')#红色
        plt.ylim((-0.4,0.4))
        plt.xlim((0,self.N_frame))
        x_ticks = np.arange(0,self.N_frame,50)
        plt.xticks(x_ticks)
        correlation_folder = save_folder+r'\\Correlation_vs_Stim_Maps'
        pp.mkdir(correlation_folder)
        all_colors = ['b','g','r','c','m','y','k','w']
  
        for i in range(len(all_keys)):#cycle all stim graphs
            if self.shuffle_flag ==True:# if true, add shuffle data in it.
                down_lim = self.Shuffled_Dictionary[all_keys[i]][0]
                shuffle_avr = self.Shuffled_Dictionary[all_keys[i]][1]
                up_lim = self.Shuffled_Dictionary[all_keys[i]][2]
            plt.fill_between(range(self.N_frame),up_lim,0,color = '#dfdfdf')
            plt.fill_between(range(self.N_frame),down_lim,0,color = '#dfdfdf')
            plt.plot(shuffle_avr,color = '#8d8d8d')
            current_plot = self.correlation_calculator(self.all_stim_graph[all_keys[i]])
            plt.plot(current_plot,color = all_colors[i],label=all_keys[i])
        plt.legend()
        plt.savefig(correlation_folder+r'\\All_plots.png')
        plt.close('all')
            
        
    def plot_target(self,**kwargs):#指定特定的序列plot在一起
        
        plot_graphs = list(kwargs.values())#Graphs need to be plotted
        plt.figure(figsize = (200,7))
        plt.title('Correlation Plot of Spon & Target Graph')
        plt.xlabel('Frames')
        plt.ylabel('Pearson R')
        plt.grid()#网格
        plt.axhline(y = 0,color = '#d62728')#红色
        plt.ylim((-0.4,0.4))
        plt.xlim((0,self.N_frame))
        x_ticks = np.arange(0,self.N_frame,50)
        plt.xticks(x_ticks)
        correlation_folder = save_folder+r'\\Correlation_vs_Stim_Maps'
        pp.mkdir(correlation_folder)
        all_colors = ['b','g','r','c','m','y','k','w']
  
        for i in range(len(plot_graphs)):#cycle all stim graphs
            
            if self.shuffle_flag ==True:# if true, add shuffle data in it.
                down_lim = self.Shuffled_Dictionary[plot_graphs[i]][0]
                shuffle_avr = self.Shuffled_Dictionary[plot_graphs[i]][1]
                up_lim = self.Shuffled_Dictionary[plot_graphs[i]][2]
            plt.fill_between(range(self.N_frame),up_lim,0,color = '#dfdfdf')
            plt.fill_between(range(self.N_frame),down_lim,0,color = '#dfdfdf')
            plt.plot(shuffle_avr,color = '#8d8d8d')
                     
            current_plot = self.correlation_calculator(self.all_stim_graph[plot_graphs[i]])
            plt.plot(current_plot,color = all_colors[i],label=plot_graphs[i])
        plt.legend()
        plt.savefig(correlation_folder+r'\\Target_Plots.png')
        plt.close('all')
    
if __name__ == '__main__':
    
    save_folder = r'E:\ZR\Data_Temp\190412_L74_LM\1-001\results'
    spike_train = pp.read_variable(save_folder+r'\spike_train_Morphology.pkl')
    graph_folder = r'E:\ZR\Data_Temp\190412_L74_LM\All-Stim-Maps\Run02'
    shuffle_flag = True# if true, annotate shuffle lines in graph.
    CF = Correlation_Functions(save_folder,graph_folder,spike_train,shuffle_flag)
    CF.stim_graph_cycle()
    CF.plot_correlation_single()
    CF.plot_all()
    all_stim_graph = CF.all_stim_graph
    #%%Manual Plot, in need of comparation
    CF.plot_target(graph_1 = 'Orien0-0',graph_2 = 'Orien90-0',graph_3 = 'Orien135-0',graph_4 ='Orien45-0' )

    