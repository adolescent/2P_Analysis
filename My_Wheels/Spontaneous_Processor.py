# -*- coding: utf-8 -*-
"""
Created on Fri May 21 15:05:52 2021

@author: ZR
"""
import OS_Tools_Kit as ot
import Filters
import pandas as pd
import numpy as np
import cv2
from sklearn import decomposition
import seaborn as sns
import matplotlib.pyplot as plt
import Graph_Operation_Kit as gt


class Spontaneous_Processor(object):
    
    name = 'Spontaneous_Processor'
    
    def __init__(self,day_folder,spon_run = 'Run001',fps = 1.301,passed_band = (0.05,0.5)):
        fp = fps/2
        if passed_band[0] != False:
            HP_par = passed_band[0]/fp
        else:
            HP_par = False
        if passed_band[1] != False:
            LP_par = passed_band[1]/fp
        else:
            LP_par = False
        self.save_folder = day_folder+r'\_All_Results\Spon_Analyze'
        ot.mkdir(self.save_folder)
        self.base_graph = cv2.imread(day_folder+r'\Global_Average.tif',-1)
        self.fps = fps
        cell_file_name = ot.Get_File_Name(day_folder,'.ac')[0]
        cell_dic = ot.Load_Variable(cell_file_name)
        all_cell_name = list(cell_dic.keys())
        spon_train_F = {}
        spon_train_dF = {}
        self.all_cell_info = {}
        for i in range(len(all_cell_name)):
            tc = cell_dic[all_cell_name[i]]
            self.all_cell_info[all_cell_name[i]] = tc['Cell_Info']
            if tc['In_Run'][spon_run]:
                c_name = tc['Name']
                c_train = tc[spon_run]['F_train']
                # do filter
                filted_c_train = Filters.Signal_Filter(c_train,filter_para = (HP_par,LP_par))
                spon_train_F[c_name] = filted_c_train
                average_F = filted_c_train.mean()
                c_dF_series = (filted_c_train-average_F)/average_F
                c_dF_series = np.clip(c_dF_series,c_dF_series.mean()-c_dF_series.std()*3,c_dF_series.mean()+c_dF_series.std()*3)
                spon_train_dF[c_name] = c_dF_series
        self.Spon_Data_Frame_Raw = pd.DataFrame(spon_train_F).T
        self.Spon_Data_Frame_centered = pd.DataFrame(spon_train_dF).T
        self.spon_cellname = list(self.Spon_Data_Frame_centered.index)
        self.frame_num = self.Spon_Data_Frame_centered.shape[1]
        
        del cell_dic
    
    def Series_Select(self,start_time,end_time,mode = 'processed'):
        if mode == 'processed':
            temp_series = self.Spon_Data_Frame_centered
        elif mode == 'raw':
            temp_series = self.Spon_Data_Frame_Raw
        start_frame = round(start_time*self.fps)
        end_frame = round(end_time*self.fps)
        if end_frame > (self.frame_num-1):
            end_frame = self.frame_num-1
            print('Index exceed, Last Frame: '+str(end_frame))
        selected_series = temp_series.iloc[:,start_frame:end_frame]
        return selected_series
        
    def Do_PCA(self,start_time = 0,end_time = 9999,plot = True,mode = 'processed'):
        '''
        Do PCA Analyze for spon series of given time.

        Parameters
        ----------
        start_time : int, optional
            Second of series ON. The default is 0.
        end_time : TYPE, optional
            Second of series OFF. The default is 9999.
        mode : 'processed' or 'raw', optional
            Which mode we use to plot PCA on. The default is 'processed'.

        Returns
        -------
        PCA_Dic : Dic
            Dictionary of PCA information.

        '''
        print('Do PCA for spontaneous cells')
        PCA_Dic = {}
        data_use = self.Series_Select(start_time, end_time,mode)
        data_for_pca = np.array(data_use).T
        pca = decomposition.PCA()
        pca.fit(data_for_pca)
        
        PCA_Dic['All_Components'] = pca.components_
        PCA_Dic['Variance_Ratio'] = pca.explained_variance_ratio_ 
        PCA_Dic['Variance'] = pca.explained_variance_
        PCA_Dic['Cell_Name_List'] = self.spon_cellname
        # plot ROC curve of PCA results.
        accumulated_ratio = np.zeros(len(PCA_Dic['Variance_Ratio']),dtype = 'f8')
        accumulated_variance = np.zeros(len(PCA_Dic['Variance']),dtype = 'f8')
        random_ratio = np.zeros(len(PCA_Dic['Variance_Ratio']),dtype = 'f8')
        for i in range(len(accumulated_ratio)-1):
            accumulated_ratio[i+1] = accumulated_ratio[i]+PCA_Dic['Variance_Ratio'][i]
            accumulated_variance[i+1] = accumulated_variance[i]+PCA_Dic['Variance'][i]
            random_ratio[i+1] = (i+1)/len(accumulated_ratio)
        PCA_Dic['Accumulated_Variance_Ratio'] = accumulated_ratio
        PCA_Dic['Accumulated_Variance'] = accumulated_variance
        
        if plot == True:
            pca_save_folder = self.save_folder+r'\PC_Graphs'
            ot.mkdir(pca_save_folder)
            for i in range(len(pca.components_[:,0])):
                visual_data,folded_map,gray_graph = self.Component_Visualize(PCA_Dic['All_Components'][i,:])
                fig = plt.figure(figsize = (15,15))
                plt.title('PC'+str(i+1),fontsize=36)
                fig = sns.heatmap(visual_data,square=True,yticklabels=False,xticklabels=False,center = 0)
                fig.figure.savefig(pca_save_folder+'\PC'+str(i+1)+'.png')
                plt.clf()
                cv2.imwrite(pca_save_folder+'\PC'+str(i+1)+'_Folded.tif',folded_map)
                cv2.imwrite(pca_save_folder+'\PC'+str(i+1)+'_Gray.jpg',gray_graph)
            
            fig,ax = plt.subplots(figsize = (8,6))
            plt.title('Accumulated Variance')
            plt.plot(range(len(accumulated_ratio)),accumulated_ratio)
            plt.plot(range(len(accumulated_ratio)),random_ratio)
            plt.savefig(pca_save_folder+'\_ROC.png')
            ot.Save_Variable(pca_save_folder, 'PCA_Dic', PCA_Dic)
            
            
        return PCA_Dic
    
    
    def Component_Visualize(self,input_component):# input component shall be a cell combination.
        
        visualize_data = np.zeros(shape = self.base_graph.shape,dtype = 'f8')
        cell_list = self.spon_cellname
        if len(input_component) != len(cell_list):
            raise IOError('Cell number mistach!')
        for i in range(len(cell_list)):
            c_cell = cell_list[i]
            c_weight = input_component[i]
            c_info = self.all_cell_info[c_cell]
            y_list,x_list = c_info.coords[:,0],c_info.coords[:,1]
            visualize_data[y_list,x_list] = c_weight
        # keep zero stable, and normalize.
        norm_visualize_graph = visualize_data/abs(visualize_data).max()
        posi_parts = norm_visualize_graph*(norm_visualize_graph>0)
        nega_parts = norm_visualize_graph*(norm_visualize_graph<0)
        # Then get folded graph.
        folded_map = cv2.cvtColor(self.base_graph,cv2.COLOR_GRAY2RGB)*0.7
        folded_map[:,:,0] += (-nega_parts)*65535
        folded_map[:,:,2] += posi_parts*65535
        folded_map = np.clip(folded_map,0,65535).astype('u2')
        gray_graph = (norm_visualize_graph*127+127).astype('u1')
        
        return visualize_data,folded_map,gray_graph
        


