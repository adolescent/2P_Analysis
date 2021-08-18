# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 14:23:45 2021

@author: ZR
"""
import pandas as pd
import numpy as np
from sklearn import decomposition
import OS_Tools_Kit as ot
import seaborn as sns
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

    '''
    # Initialization
    print('We do PCA here.')
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
    
    return components,PCA_info



def Compoment_Visualize(components,all_cell_dic,output_folder,graph_shape = (512,512)):
    
    '''
    Visualize component 

    Parameters
    ----------
    components : (pd Frame)
        Data Frame of PCA components.
    all_cell_dic : (dic)
        Read in '.ac' file. This is used to get cell location.
    output_folder : (str)
        PC components subfolder will be put in this path.

    Returns
    -------
    bool
        Indicate we plot graphs here.

    '''
    
    all_cell_info = {}
    acn = list(all_cell_dic.keys())
    for i,ccn in enumerate(acn):
        all_cell_info[ccn] = all_cell_dic[ccn]['Cell_Info']
    all_PC_names = components.columns.tolist()
    
    # plot each graphs, origin data, and save.
    PC_Graph_Data = {}
    PCA_folder = output_folder+r'\PCA_Graphs'
    ot.mkdir(PCA_folder)
    for i,current_PC in enumerate(all_PC_names):
        c_component = components[current_PC]
        c_graph = np.zeros(shape = graph_shape,dtype = 'f8')
        for j,ccn in enumerate(acn):
            c_cell_info = all_cell_info[ccn]
            y_list,x_list = c_cell_info.coords[:,0],c_cell_info.coords[:,1]
            c_graph[y_list,x_list] = c_component[ccn]
        PC_Graph_Data[current_PC] = c_graph
        fig = plt.figure(figsize = (15,15))
        plt.title(current_PC,fontsize=36)
        fig = sns.heatmap(c_graph,square=True,yticklabels=False,xticklabels=False,center = 0)
        fig.figure.savefig(PCA_folder+r'\\'+current_PC+'.png')
        plt.clf()
    #%% Fit PCA, get fitted results
    fitted_weights = None
    return PC_Graph_Data,fitted_weights
