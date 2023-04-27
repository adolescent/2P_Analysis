# -*- coding: utf-8 -*-
"""
Created on Fri May 27 17:16:18 2022

@author: adolescent
"""

import pandas as pd
import numpy as np
from sklearn import decomposition
import OS_Tools_Kit as ot
import seaborn as sns
import matplotlib.pyplot as plt
from Series_Analyzer.Preprocessor_Cai import Pre_Processor_Cai
import List_Operation_Kit as lt
from Decorators import Timer
import warnings
import seaborn as sns
import cv2
import matplotlib.pyplot as plt

def Do_PCA(input_frame):
    '''
    Input cell data frames, return PCA components and PCA variance accomulation,

    Parameters
    ----------
    inpu_frame : (pd Frame)
        Cell data frame(row as a cell, column as a graph).

    Returns
    -------
    components : (pd Frame)
        PCA components(row as a cell, column as a component).
    PCA_info : (Dic)
        Information of PCA result.
    fitted_weights : (pd Frame)
        

    '''
    # Initialization
    print('We do PCA here.')
    warnings.filterwarnings("ignore")
    all_cell_name = input_frame.index.tolist()
    components = pd.DataFrame(index = all_cell_name)
    PCA_info = {}
    # Do PCA
    data_for_pca = np.array(input_frame).T
    pca = decomposition.PCA()
    pca.fit(data_for_pca)
    # Fill in component frames
    all_components = pca.components_
    for i in range(all_components.shape[0]):
        c_name = 'PC'+ot.Bit_Filler(i+1,bit_num = 3)
        c_components = all_components[i,:]
        components[c_name] = c_components
    PCA_info['Variance_Ratio'] = pca.explained_variance_ratio_ 
    PCA_info['Variance'] = pca.explained_variance_
    # Get accumulated variance & accumulated ratio.
    PC_num = len(pca.explained_variance_ratio_ )
    accu_ratio = [0]
    accu_var = [0]
    for i in range(PC_num):
        accu_ratio.append(accu_ratio[i]+pca.explained_variance_ratio_[i])
        accu_var.append(accu_var[i]+pca.explained_variance_[i])
    PCA_info['Accumulated_Variance_Ratio'] = accu_ratio
    PCA_info['Accumulated_Variance'] = accu_var
    # Fit PCA, get fitted results
    raw_fitted_weight = pca.transform(data_for_pca)
    column_names = list(range(1,1+raw_fitted_weight.shape[1]))
    for i,c_column in enumerate(column_names):
        column_names[i] = 'PC'+ot.Bit_Filler(c_column,3)
    fitted_weights = pd.DataFrame(raw_fitted_weight,columns = column_names)
    return components,PCA_info,fitted_weights


def Comp_Visualize_Cai(c_comp,all_cell_dic,show = True):
    
    acn = list(all_cell_dic.keys())
    graph_shape = all_cell_dic[acn[0]]['Cell_Mask'].shape
    c_PC_graph = np.zeros(shape = graph_shape,dtype = 'f8')
    for i,cc in enumerate(acn):# cv2 will load frame in sequence x,y.
        cc_loc = (all_cell_dic[cc]['Cell_Loc'].astype('i4')[1],all_cell_dic[cc]['Cell_Loc'].astype('i4')[0])
        c_resp = c_comp.loc[cc]
        c_PC_graph = cv2.circle(c_PC_graph,cc_loc,4,c_resp,-1)
    # if show, plot in sns heatmap.
    if show:
        sns.heatmap(c_PC_graph,center = 0,square = True,yticklabels=False,xticklabels=False)        
    return c_PC_graph

@Timer
def One_Key_PCA(day_folder,runname,
                tag = 'Spon_Before',subfolder = '_CAIMAN',
                start_frame = 0,end_frame = 99999):
    workpath = ot.join(day_folder,subfolder)
    all_cell_dic = ot.Load_Variable(workpath,'All_Series_Dic.pkl')
    save_folder = ot.join(workpath,tag+'_PCA')
    _ = ot.mkdir(save_folder)
    data_frame = Pre_Processor_Cai(day_folder,runname,subfolder,start_frame,end_frame,passed_band=(0.005,0.3),order = 7)
    components,PCA_info,fitted_weights = Do_PCA(data_frame)
    ot.Save_Variable(save_folder, 'All_PC_Components', components)
    ot.Save_Variable(save_folder, 'All_PC_Info', PCA_info)
    ot.Save_Variable(save_folder, 'fitted_weights', fitted_weights)
    PC_name = list(components.columns)
    for i,c_PC in enumerate(PC_name):
        c_comp = components.loc[:,c_PC]
        c_graph = Comp_Visualize_Cai(c_comp,all_cell_dic,show = False)
        fig = plt.figure(figsize = (15,15))
        plt.title(c_PC,fontsize=36)
        fig = sns.heatmap(c_graph,square=True,yticklabels=False,xticklabels=False,center = 0)
        fig.figure.savefig(ot.join(save_folder,c_PC)+'.png')
        plt.clf()
        plt.close()
    
    
    
    return components,PCA_info,fitted_weights

def PC_Reduction(input_frame,PC_Range = [2,100]):
    # make sure input frame have struction N_sample*M_Cell
    print('Generating PC reducted components.')
    reducted_frame = np.zeros(shape = input_frame.shape,dtype = 'f8')
    pca = decomposition.PCA(n_components = PC_Range[1])
    pca.fit(input_frame)
    all_comps = pca.components_ # N_PCNum*M_Dims, transfer matrix.
    all_reps = pca.transform(input_frame) # N Samples*M PC Nums, representation in new space.
    used_PC_lists = list(range(PC_Range[0],PC_Range[1]+1))
    # cycle used PC return data.
    sample_num = reducted_frame.shape[0]
    for j in range(sample_num):
        c_series = np.zeros(reducted_frame.shape[1],dtype = 'f8')
        for i,cc in enumerate(used_PC_lists):
            c_weight = all_reps[j,cc-1]
            c_series += c_weight*all_comps[cc-1,:]
        reducted_frame[j,:] = c_series
    
    return reducted_frame


#%% test run
if __name__ == '__main__':
    
    day_folder = r'D:\Test_Data\2P\220421_L85'
    runname = '1-001'
    comp,info,weight = One_Key_PCA(day_folder, runname,start_time= 8000)
    