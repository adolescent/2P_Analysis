# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 11:33:08 2019

@author: ZR
This tool will calculate the person correlation between a stimulus map and a set of clustered datas
"""
import General_Functions.my_tools as pp
import numpy as np
import cv2
import scipy.stats
import matplotlib.pyplot as plt

class Correlation_Description(object):
    
    name = 'Correlation calculation'
    
    def __init__(self,save_folder,target_graph,clustered_data,correlation_name):
        
        self.target_graph = target_graph
        self.clustered_data = clustered_data
        self.correlation_name = correlation_name
        self.correlation_folder = save_folder+r'\\Correlation_Folder'
        pp.mkdir(self.correlation_folder)
        
    def calculation_unit(self,i):#计算单元，计算当前component i 和目标图的相似度。
        
        temp_component = self.clustered_data[i,:]
        r,_ = scipy.stats.pearsonr(self.target_graph[:,0],temp_component)
        return r
    
    def correlation_calculation(self):#这里把所有的相关数值计算出来，然后plot分布
        
        self.r_value = np.zeros(np.shape(clustered_data)[0],dtype = np.float64)
        for i in range(len(self.r_value)):
            self.r_value[i] = self.calculation_unit(i)
        pp.save_variable(self.r_value,self.correlation_folder+r'\\R_Values_'+self.correlation_name+'.pkl')
        
    def correlation_discription(self):#对相关做描述统计，包括绘图，均值，标准差，超过2/3std的图等。 
        plt.figure(figsize = (20,8))
        plt.hist(self.r_value,bins = 50)
        plt.title('Correlation Distribution of '+self.correlation_name)
        plt.xlabel('Pearson r')
        plt.ylabel('Counts')
        #plt.annotate(('std = '+str(self.r_value.std())),xy = (self.r_value.max(),0),xytext = (self.r_value.max(),-8))
        plt.savefig(self.correlation_folder+r'\\'+self.correlation_name+'.png')
        plt.close('all')
        #除了直方图之外，也写一个txt文本包含主要信息
        temp_std = self.r_value.std()
        temp_mean = self.r_value.mean()
        thres = 2*temp_std+temp_mean#作为相关图的阈值
        find_list = abs(self.r_value)>thres
        similar_ids = list(np.where(find_list == True)[0])#这就是相似的成分ID
        f = open(self.correlation_folder+r'\\'+self.correlation_name+'_Discription.txt','w')
        f.write('General Discriptions:\n')
        f.write('Correlations have std = '+str(temp_std)+', mean = '+str(temp_mean)+'\n')
        f.write('Max correlation = '+str(self.r_value.max())+', Minimum = '+str(self.r_value.min())+'\n')
        f.write('Correlation components above 2std are as below:\n')
        for i in range(len(similar_ids)):
            f.write('Component '+str(similar_ids[i]+1)+', Pearson r = '+str(self.r_value[similar_ids[i]])+'\n')
        f.close()
        
        
        
        
        
if __name__ == '__main__':
    #写成批处理形式
    save_folder = r'E:\ZR\Data_Temp\190412_L74_LM\1-001\results'
    graph_folder = r'E:\ZR\Data_Temp\190412_L74_LM\All-Stim-Maps\Run02'
    clustered_data = pp.read_variable(save_folder+r'\Mini_KMeans_Data.pkl')
    all_graph_name = pp.file_name(graph_folder,'.pkl')
    cluster_type = 'KMeans'
    for i in range(len(all_graph_name)):
        correlation_name = cluster_type+'_vs_'+all_graph_name[i].split('\\')[-1][:-4]
        target_graph = pp.read_variable(all_graph_name[i])
        CD = Correlation_Description(save_folder,target_graph,clustered_data,correlation_name)
        CD.correlation_calculation()
        CD.correlation_discription()
        
        
# =============================================================================
#     target_graph = pp.read_variable(r'E:\ZR\Data_Temp\190412_L74_LM\All-Stim-Maps\Run02\Orien90-0_Cells.pkl')
#     clustered_data = pp.read_variable(save_folder+r'\ICAed_Data.pkl')
#     correlation_name = 'PCA_vs_Orien90-0'
#     CD = Correlation_Description(save_folder,target_graph,clustered_data,correlation_name)
#     CD.correlation_calculation()
#     CD.correlation_discription()
#     test = pp.file_name(r'E:\ZR\Data_Temp\190412_L74_LM\All-Stim-Maps\Run02','.pkl')
# =============================================================================
