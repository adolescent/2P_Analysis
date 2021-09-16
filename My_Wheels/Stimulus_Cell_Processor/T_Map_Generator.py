# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 16:26:15 2021

@author: ZR

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import OS_Tools_Kit as ot
import Graph_Operation_Kit as gt
from Analyzer.Statistic_Tools import T_Test_Pair
from Standard_Parameters.Sub_Graph_Dics import Sub_Dic_Generator


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


def One_Key_T_Maps(day_folder,runname,runtype = 'OD_2P',para = None,
                   p_thres = 0.05,used_frame = [4,5]):
    
    save_folder = day_folder+r'\_All_Results\\'+runtype+'_t_Maps'
    ot.mkdir(save_folder)
    all_t_map_info = {}
    all_t_map_info['D_maps'] = {}
    all_t_map_info['p_maps'] = {}
    all_t_map_info['cell_info'] = {}
    cd_name = ot.Get_File_Name(day_folder,'.ac')[0]
    all_cell_dic = ot.Load_Variable(cd_name)
    sub_dics = Sub_Dic_Generator(runtype,para)
    all_graph_name = list(sub_dics.keys())
    for i,c_graph in enumerate(all_graph_name):
        A_id,B_id = sub_dics[c_graph]
        c_D_map,c_p_map,c_response = T_Map_Core(all_cell_dic, runname, A_id, B_id,p_thres,used_frame)
        all_t_map_info['D_maps'][c_graph] = c_D_map
        all_t_map_info['p_maps'][c_graph] = c_p_map
        all_t_map_info['cell_info'][c_graph] = c_response
    # Visualize cells in all t graphs
    average_graph = cv2.imread(day_folder+'\\Global_Average.tif',1)
    for i,c_graph in enumerate(all_graph_name):
        c_D_map = all_t_map_info['D_maps'][c_graph]
        # First standard D map
        fig = plt.figure(figsize = (15,15))
        plt.title(c_graph+' D Map',fontsize=36)
        fig = sns.heatmap(c_D_map,square=True,yticklabels=False,xticklabels=False,center = 0)
        fig.figure.savefig(save_folder+r'\\'+c_graph+'_D_Map.png')
        plt.close()
        # Then get folded map.
        folded_map = average_graph*0.7
        normed_c_D_map = c_D_map/abs(c_D_map).max()
        posi_part = normed_c_D_map*(normed_c_D_map>0)
        nega_part = -normed_c_D_map*(normed_c_D_map<0)
        folded_map[:,:,2] += posi_part*255
        folded_map[:,:,0] += nega_part*255
        folded_map = np.clip(folded_map,0,255).astype('u1')
        gt.Show_Graph(folded_map, c_graph+'_Folded', save_folder,0)
    ot.Save_Variable(save_folder, 'All_T_Info', all_t_map_info)
    return all_t_map_info