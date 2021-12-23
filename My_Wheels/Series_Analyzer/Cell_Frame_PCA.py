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
from Series_Analyzer.Spontaneous_Preprocessing import Pre_Processor
import List_Operation_Kit as lt
from Decorators import Timer


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
        cells_in_PC = c_component.index.tolist()
        for j,ccn in enumerate(cells_in_PC):
            c_cell_info = all_cell_info[ccn]
            y_list,x_list = c_cell_info.coords[:,0],c_cell_info.coords[:,1]
            c_graph[y_list,x_list] = c_component[ccn]
        PC_Graph_Data[current_PC] = c_graph
        fig = plt.figure(figsize = (15,15))
        plt.title(current_PC,fontsize=36)
        fig = sns.heatmap(c_graph,square=True,yticklabels=False,xticklabels=False,center = 0)
        fig.figure.savefig(PCA_folder+r'\\'+current_PC+'.png')
        plt.clf()
        plt.close()
    return PC_Graph_Data

def One_Key_PCA(day_folder,runname,tag = 'Spon_Before',
                start_time = 0,end_time = 99999):
    
    '''
    One key generate PCA graphs. Most function generated.

    Parameters
    ----------
    day_folder : (str)
        Day of run folder.
    runname : (str)
        Runname of run to be processed. e.g.'Run001'
    tag : (str),optional
        Tag of PCA we 
    start_time : (int),optional
        Seconds of frame start. Can be used to ignore initial supression.
    end_time : (int),optional
        Seconds of frame end.
        

    Returns
    -------
    components : (pd Frame)
        DESCRIPTION.
    PCA_info : (Dic)
        DESCRIPTION.
    fitted_weights : (pd Frame)
        DESCRIPTION.

    '''
    save_folder = day_folder+r'\_All_Results\PCA_'+tag
    _ = ot.mkdir(save_folder,mute = True)
    # First, calculate PC components
    all_cell_dic_folder = ot.Get_File_Name(day_folder,'.ac')[0]
    all_cell_dic = ot.Load_Variable(all_cell_dic_folder)
    data_frame = Pre_Processor(day_folder,runname,start_time,end_time,passed_band=(0.005,0.3),order = 7)
    components,PCA_info,fitted_weights = Do_PCA(data_frame)
    ot.Save_Variable(save_folder, 'All_PC_Components', components)
    ot.Save_Variable(save_folder, 'All_PC_Info', PCA_info)
    ot.Save_Variable(save_folder, 'fitted_weights', fitted_weights)
    # Second, Generate PCA graphs.
    _ = Compoment_Visualize(components,all_cell_dic,save_folder)
    return components,PCA_info,fitted_weights



@Timer
def PCA_Regression(PC_components,PC_info,fitted_weights,ignore_PC = [1],var_ratio = 0.95):
    '''
    Regress specific PC component, used for global detraction.
    

    Parameters
    ----------
    PC_components : (pd Frame)
        Component of each PCA.
    PC_info : (dic)
        Dictionary of PCA information.
    fitted_weights : (pd Frame)
        Fitted PC weights of all frames. CORE input.
    ignore_PC : (list), optional
        List of PC need to be ignored. Just input number. The default is [1].
    var_ratio : (float), optional
        Propotion of variation we use. Least important PCs will be ignored. The default is 0.9.

    Returns
    -------
    regressed_data_trains : (pd Frame)
        Trains of regressed data. Cell trains.

    '''
    # get PC frames here.
    acc_ratio = np.array(PC_info['Accumulated_Variance_Ratio'])
    last_pc = np.where(acc_ratio>var_ratio)[0][0]+1
    all_pc_name = []
    for i in range(1,last_pc+1):
        all_pc_name.append('PC'+ot.Bit_Filler(i,3))
    ignored_pc_name = []
    for i,ig_pc in enumerate(ignore_PC):
        ignored_pc_name.append('PC'+ot.Bit_Filler(ig_pc,3))
    used_pc_name = lt.List_Subtraction(all_pc_name, ignored_pc_name)
    # get regressed data.
    acn = PC_components.index
    all_frame_name = fitted_weights.index
    regressed_frame = pd.DataFrame(index = all_frame_name,columns = acn)
    for i,c_frame in enumerate(all_frame_name):
        c_weight = fitted_weights.loc[c_frame]
        c_regressed_graph = np.zeros(fitted_weights.shape[1])
        for j,c_pc in enumerate(used_pc_name):
            c_regressed_graph += PC_components[c_pc]*c_weight[c_pc]
        c_regressed_graph = np.array(c_regressed_graph)
        regressed_frame.iloc[c_frame,:] = c_regressed_graph
    regressed_frame = regressed_frame.T # keep shape the same.
    return regressed_frame

