# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 15:28:39 2019

@author: ZR
Many decomposition method provided by scikit-learn, not sure whether they are useful.

"""

import General_Functions.my_tools as pp
import sklearn.decomposition as skdecomp
import numpy as np
import cv2
import sklearn.cluster as skcluster


class Alchemy_Cells(object):
    
    name = 'Do ICA Analysis and many other alchemys to spontaneous cell datas'
    
    def __init__(self,save_folder,spike_train_name,cell_group_name):
        
        self.save_folder = save_folder
        spike_train = pp.read_variable(self.save_folder+r'\\'+spike_train_name)
        self.cell_group = pp.read_variable(self.save_folder+r'\\'+cell_group_name)
        self.vector = spike_train.T#转置矩阵，并记作vector
        
    def preprocessing(self):
        
        #除了非负矩阵分解（NMF）方法之外，其他方法都需要使用中心化之后的数据= =
        n_samples,n_features = self.vector.shape
        self.vector = pp.normalization(self.vector)
        # 全局中心化，每个像素减去跨run的均值
        self.vector_centered = self.vector - self.vector.mean(axis=0)
        # 局域中心化，把每张图减掉自己的均值。
        self.vector_centered -= self.vector_centered.mean(axis=1).reshape(n_samples, -1)#必须要reshape，不然行列不对
    
    def cell_graph_plot(self,name,features):#注意输入格式，需要features第一个维度是样本数
        graph_folder = save_folder+r'\\'+name
        pp.mkdir(graph_folder)
        for i in range(np.shape(features)[0]):
            temp_graph = np.ones(shape = (512,512),dtype = np.uint8)*127#基底图
            temp_feature = features[i]
            if temp_feature.max() !=0:
                norm_temp_feature = (temp_feature/(abs(temp_feature).max())+1)/2#保持0不变，并拉伸至0-1的图
            else:
                norm_temp_feature = (temp_feature+1)/2
            
            for j in range(len(self.cell_group)):#循环细胞
                x_list,y_list = pp.cell_location(self.cell_group[j])
                temp_graph[y_list,x_list] = norm_temp_feature[j]*255
            temp_graph_name = graph_folder+r'\\'+name+str(i+1)
            current_graph_labled = cv2.cvtColor(np.uint8(temp_graph),cv2.COLOR_GRAY2BGR)
            cv2.imwrite(temp_graph_name+'.png',np.uint8(cv2.resize(current_graph_labled,(1024,1024))))
            pp.show_cell(temp_graph_name+'.png',self.cell_group)
        
    
    
    
    def PCA(self,N_component,whiten_flag):#注意，N_component = None即保留了全部的PCA成分，数目就是维度的数目
         
        PCA_calculator = skdecomp.PCA(n_components = N_component,whiten = whiten_flag)
        self.PCAed_data = PCA_calculator.fit(self.vector_centered)
        all_PCs = self.PCAed_data.components_#这个输出归一化过，所有的PC模长都是1
        PC_variance_ratio = self.PCAed_data.explained_variance_ratio_
        pp.save_variable(all_PCs,save_folder+r'\\PCAed_Data.pkl')
        #注意调用的时候.components_为主成分合集，第一个维度是components数，第二个是feature，所以每一横行是一个图。
        ####之后把PCA成分可视化保存。
        print('PCA calculation done, generating graphs')
        self.cell_graph_plot('PCA',all_PCs)
        
        f = open(self.save_folder+r'\PCA\Frame_Count.txt','w')
        for i in range(len(PC_variance_ratio)):
            f.write('PC:'+str(i+1)+' expianed '+str(PC_variance_ratio[i])+'of all variance.\n')
        f.close()
        
    def ICA(self,N_component):
        ICA_calculator = skdecomp.FastICA(N_component,max_iter=1500,tol=0.03,whiten=True)
        self.ICAed_data = ICA_calculator.fit(self.vector_centered)
        all_ICs = self.ICAed_data.components_
        pp.save_variable(all_ICs,save_folder+r'\\ICAed_Data.pkl')
        print('ICA calculation done, generating graphs')
        self.cell_graph_plot('ICA',all_ICs)
        
    def NMF(self,N_component):#Non-Negative Matrix Factorization
        NMF_calculator = skdecomp.NMF(N_component,max_iter=1500,tol = 0.0001)
        self.NMFed_data = NMF_calculator.fit(self.vector)
        all_NMFs = self.NMFed_data.components_
        pp.save_variable(all_NMFs,save_folder+r'\\NMFed_Data.pkl')
        print('NMF calculation done, generating graphs')
        self.cell_graph_plot('NMF',all_NMFs)
        
    def MiniBach_SparcePCA(self,N_component):
        MiniPCA_calculator = skdecomp.MiniBatchSparsePCA(N_component,batch_size=15,normalize_components=True)
        self.MiniPCs = MiniPCA_calculator.fit(self.vector_centered)
        all_MiniPCs = self.MiniPCs.components_
        pp.save_variable(all_MiniPCs,save_folder+r'\\MiniPCed_Data.pkl')
        print('MiniBach Sparce PCA done, generating graphs')
        self.cell_graph_plot('MINIPCA',all_MiniPCs)
    
    def MiniBach_DictionaryLearning(self,N_component):
        MiniDL_calculator = skdecomp.MiniBatchDictionaryLearning(N_component,batch_size = 25)
        self.MiniDLs = MiniDL_calculator.fit(self.vector_centered)
        all_MiniDLs = self.MiniDLs.components_
        pp.save_variable(all_MiniDLs,save_folder+r'\\Dictionary_Learning_Data.pkl')
        print('MiniBach Dictionary Learning Done, generating graphs')
        self.cell_graph_plot('Dictionary_Learning',all_MiniDLs)
    def MiniBach_KMeans(self,N_component):
        MiniKM_calculator = skcluster.MiniBatchKMeans(N_component,batch_size = 50)
        self.MiniKM = MiniKM_calculator.fit(self.vector_centered)
        all_MiniKMs = self.MiniKM.cluster_centers_
        pp.save_variable(all_MiniKMs,save_folder+r'\\Mini_KMeans_Data.pkl')
        print('Mini bach K-means done, generating graphs')
        self.cell_graph_plot('MiniBach_KMeans',all_MiniKMs)
        
    def FactorAnalysis(self,N_component):
        Factor_Analysis_calculator = skdecomp.FactorAnalysis(N_component)
        self.Analysized_Factors = Factor_Analysis_calculator.fit(self.vector_centered)
        all_FAs = self.Analysized_Factors.components_
        pp.save_variable(all_FAs,save_folder+r'\\Factor_Analysis_Data.pkl')
        print('Factor Analysis done, generating graphs')
        self.cell_graph_plot('Analyzed Factors',all_FAs)
    
if __name__ == '__main__':
    save_folder = r'E:\ZR\Data_Temp\190412_L74_LM\1-001\results'
    spike_train_name = 'spike_train_Morphology_filtered.pkl'
    cell_group_name = 'Cell_Groups_Morphology.pkl'
    AC = Alchemy_Cells(save_folder,spike_train_name,cell_group_name)
    AC.preprocessing()
    #AC.PCA(None,False)
    #AC.ICA(100)
    #AC.NMF(100)
    #AC.MiniBach_SparcePCA(10)
    #AC.MiniBach_DictionaryLearning(100)
    #AC.MiniBach_KMeans(100)
    AC.FactorAnalysis(50)
    #%%
   # a = AC.PCAed_data.components_2
    #b = AC.PCAed_data.explained_variance_ratio_
    