# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 11:34:18 2021

@author: ZR
"""

import numpy as np
import seaborn as sns

def Single_Comp_Visualize(all_cell_dic,input_frame,shape = (512,512)):
    '''
    Plot Single Component graphs.

    Parameters
    ----------
    all_cell_dic : (dic)
        All Cell Dics.
    input_frame : (pd Series)
        Values series of cell graph. Cell ID need to be included.
    shape : (turple), optional
        Shape of output graph. The default is (512,512).

    Returns
    -------
    graph : (2D_array)
        Visualized cell graph.

    '''
    graph = np.zeros(shape = shape,dtype = 'f8')
    used_cell_name = list(input_frame.index)
    for i,cc in enumerate(used_cell_name):
        c_dic = all_cell_dic[cc]
        cc_info = c_dic['Cell_Info']
        y_list,x_list = cc_info.coords[:,0],cc_info.coords[:,1]
        graph[y_list,x_list] = input_frame[cc]
    sns.heatmap(graph,center = 0,square = 1, xticklabels=False, yticklabels=False)
    return graph
