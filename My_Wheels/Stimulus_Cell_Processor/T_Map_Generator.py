# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 16:26:15 2021

@author: ZR

"""
import numpy as np
import pandas as pd
from Analyzer.Statistic_Tools import T_Test_Pair


def T_Map_Core(all_cell_dic,runname,
               A_ID_lists,B_ID_lists,
               p_thres = 0.05,used_frame = [4,5]):
    '''
    Generate A-B t map from cell data.

    Parameters
    ----------
    all_cell_dic : (dic)
        All Cell Data, usually from '.ac' file.
    A_ID_lists : (list)
        List of A conditions.
    B_ID_lists : (list)
        List of B conditions.
    p_thres : (float), optional
        P threshold of t significant. The default is 0.05.
    used_frame : (list), optional
        List of frame used for calculation. The default is [4,5].

    Returns
    -------
    D_map_raw : (2D Array)
        Raw t data matrix. 
    p_map : (2D Array)
        p value data matrix.
    used_cell_response : (pd Frame)
       Raw t value of different cells.can be used directly.

    '''
    all_cell_name = list(all_cell_dic.keys())
    used_cells_CR_dic = {}
    for i,ccn in enumerate(all_cell_name):
        if all_cell_dic[ccn]['In_Run'][runname]:
            used_cells_CR_dic[ccn] = all_cell_dic[ccn][runname]['CR_Train']
    # calculate t value of cells first.
    used_all_cell_name = list(used_cells_CR_dic.keys())
    used_cell_response = pd.DataFrame(index = ['t','p','CohenD'])
    for i,ccn in enumerate(used_all_cell_name):
        c_CR = used_cells_CR_dic[ccn]
        A_responses = c_CR[A_ID_lists[0]]
        for i,c_A in enumerate(A_ID_lists):
            single_cond_response = c_CR[c_A]
            if i >0:
                A_responses = np.vstack((A_responses,single_cond_response))
        B_responses = c_CR[B_ID_lists[0]]
        for i,c_B in enumerate(B_ID_lists):
            single_cond_response = c_CR[c_B]
            if i>0:
                B_responses = np.vstack((B_responses,single_cond_response))
        A_ON_data = A_responses[:,used_frame].flatten()
        B_ON_data = B_responses[:,used_frame].flatten()
        c_cell_t,c_cell_p,c_cell_D = T_Test_Pair(A_ON_data, B_ON_data)
        used_cell_response[ccn] = [c_cell_t,c_cell_p,c_cell_D]
    # get visualized graph from given 
    graph_shape = all_cell_dic[all_cell_name[0]]['Cell_Info']._label_image.shape
    D_map_raw = np.zeros(shape = graph_shape,dtype = 'f8')
    p_map = np.zeros(shape = graph_shape,dtype = 'f8')
    for i,c_cell in enumerate(used_all_cell_name):
        c_cell_info = all_cell_dic[c_cell]['Cell_Info']
        y,x = c_cell_info.coords[:,0],c_cell_info.coords[:,1]
        p_map[y,x] = used_cell_response.loc['p',c_cell]
        if used_cell_response.loc['p',c_cell]<p_thres:
            D_map_raw[y,x] = used_cell_response.loc['CohenD',c_cell]
    
    return D_map_raw,p_map,used_cell_response


def One_Key_T_Maps():
    pass