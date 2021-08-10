# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 14:23:45 2021

@author: ZR
"""
import pandas as pd
import numpy as np
from sklearn import decomposition
import OS_Tools_Kit as ot


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
        
        
    return components,PCA_info



def Compoment_Visualize(components,all_cell_dic,output_folder):
    '''
    Visualize component 

    Parameters
    ----------
    components : TYPE
        DESCRIPTION.
    all_cell_dic : TYPE
        DESCRIPTION.
    output_folder : TYPE
        DESCRIPTION.

    Returns
    -------
    bool
        DESCRIPTION.

    '''
    pass
    return True
