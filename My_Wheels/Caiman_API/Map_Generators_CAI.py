# -*- coding: utf-8 -*-
"""
Created on Mon May 16 13:04:25 2022

@author: adolescent
"""



import os
from Decorators import Timer
import time
import Graph_Operation_Kit as gt
import OS_Tools_Kit as ot
import os
import List_Operation_Kit as lt
from Caiman_API.Pack_Graphs import Graph_Packer
import bokeh.plotting as bpl
from tqdm import tqdm
import cv2
import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params
from caiman.utils.utils import download_demo
from caiman.utils.visualization import plot_contours, nb_view_patches, nb_plot_contour
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import skimage.io
import pandas as pd
from caiman.source_extraction.cnmf.cnmf import load_CNMF
from caiman.source_extraction.cnmf.params import CNMFParams
from Analyzer.Statistic_Tools import T_Test_Welch,T_Test_Ind
import seaborn as sns
from Standard_Parameters.Sub_Graph_Dics import Sub_Dic_Generator
#%%

def T_Map_Core(Cell_Cond_Response,runname,all_cell_dic,
               A_ID_lists,B_ID_lists,
               p_thres = 0.05,used_frame = [4,5]):
    
    
    all_cell_name = list(Cell_Cond_Response.keys())
    used_cells_CR_dic = {}
    for i,ccn in enumerate(all_cell_name):
        used_cells_CR_dic[ccn] = Cell_Cond_Response[ccn][runname]
    # calculate t value of cells first.
    used_all_cell_name = list(used_cells_CR_dic.keys())
    used_cell_response = pd.DataFrame(index = ['t','p','CohenD'])
    for i,ccn in enumerate(used_all_cell_name):
        c_CR = used_cells_CR_dic[ccn]
        A_responses = c_CR[A_ID_lists[0]]
        for i,c_A in enumerate(A_ID_lists):
            single_cond_response = c_CR[c_A]
            if i >0:
                A_responses = np.hstack((A_responses,single_cond_response))
        B_responses = c_CR[B_ID_lists[0]]
        for i,c_B in enumerate(B_ID_lists):
            single_cond_response = c_CR[c_B]
            if i>0:
                B_responses = np.hstack((B_responses,single_cond_response))
        A_ON_data = A_responses[used_frame,:].flatten()
        B_ON_data = B_responses[used_frame,:].flatten()
        if len(A_ON_data) == len(B_ON_data):
            c_cell_t,c_cell_p,c_cell_D = T_Test_Ind(A_ON_data, B_ON_data)
        else:
            c_cell_t,c_cell_p,c_cell_D = T_Test_Welch(A_ON_data, B_ON_data)
        used_cell_response[ccn] = [c_cell_t,c_cell_p,c_cell_D]
    # get visualized graph from given 
    graph_shape = all_cell_dic[used_all_cell_name[0]]['Cell_Mask'].shape
    D_map_raw = np.zeros(shape = graph_shape,dtype = 'f8')
    p_map = np.zeros(shape = graph_shape,dtype = 'f8')
    for i,c_cell in enumerate(used_all_cell_name):
        c_mask = all_cell_dic[c_cell]['Cell_Mask']
        clipped_c_mask = c_mask/c_mask.max()
        #clipped_c_mask = c_mask
        p_map += clipped_c_mask*used_cell_response.loc['p',c_cell]
        if used_cell_response.loc['p',c_cell]<p_thres:
            D_map_raw += clipped_c_mask*used_cell_response.loc['t',c_cell]
    # Plot circle map for visualization.
    colored_circle_map = np.zeros(shape = graph_shape,dtype = 'f8')
    sized_circle_map = np.zeros(shape = graph_shape,dtype = 'f8')
    normed_response = used_cell_response.loc['t']/abs(used_cell_response.loc['t']).max()
    # Colored circle map represent response as color.
    for i,cc in enumerate(used_all_cell_name):
        if used_cell_response.loc['p',cc] < p_thres:# as significant cell
            cc_loc = (all_cell_dic[cc]['Cell_Loc'].astype('i4')[1],all_cell_dic[cc]['Cell_Loc'].astype('i4')[0])
            colored_circle_map = cv2.circle(colored_circle_map,cc_loc,4,used_cell_response.loc['t',cc],-1)
    # sized map use circle size as response.
    response_std = normed_response.std()
    response_mean = normed_response.mean()
    clipped_response = normed_response.clip(response_mean-2*response_std,response_mean+2*response_std)
    clipped_response = clipped_response/abs(clipped_response).max()
    sized_response = (clipped_response*10).astype('i4')
    for i,cc in enumerate(used_all_cell_name):
        if used_cell_response.loc['p',cc] < p_thres:# as significant cell
            cc_loc = (all_cell_dic[cc]['Cell_Loc'].astype('i4')[1],all_cell_dic[cc]['Cell_Loc'].astype('i4')[0])
            if sized_response.loc[cc]>0:
                sized_circle_map = cv2.circle(sized_circle_map,cc_loc,int(sized_response.loc[cc]),1,-1)
            elif sized_response.loc[cc]<0:
                sized_circle_map = cv2.circle(sized_circle_map,cc_loc,int(sized_response.loc[cc])*-1,-1,-1)
    return D_map_raw,p_map,colored_circle_map,sized_circle_map,used_cell_response




def One_Key_T_Map(day_folder,runname,run_type,
                  subfolder = '_CAIMAN',para = 'Default',
                  p_thres = 0.05,used_frame = [4,5]):
    
    

    workpath = ot.join(day_folder,subfolder)
    global_average = cv2.imread(ot.join(workpath,'Global_Average_cai.tif'),-1)
    all_cell_dic = ot.Load_Variable(workpath,'All_Series_Dic.pkl')
    cell_condition_response = ot.Load_Variable(workpath,'Cell_Condition_Response.pkl')
    sub_dic = Sub_Dic_Generator(mode = run_type,para = para)
    graph_path = ot.join(workpath,runname+'_T_Maps')
    ot.mkdir(graph_path)
    all_map_name = list(sub_dic.keys())
    all_cell_response = {}
    for i,map_name in enumerate(all_map_name):
        [A_lists,B_lists] = sub_dic[map_name]
        c_D_map,_,c_circle_map,c_size_map,c_cell_resp = T_Map_Core(cell_condition_response, runname, all_cell_dic, A_lists, B_lists)
        # Save cell response first.
        all_cell_response[map_name] = c_cell_resp
        # Save D Map
        fig = plt.figure(figsize = (15,15))
        plt.title(map_name+' t Map',fontsize=36)
        fig = sns.heatmap(c_D_map,square=True,yticklabels=False,xticklabels=False,center = 0)
        fig.figure.savefig(graph_path+r'\\'+map_name+'_t_Map.png')
        plt.close()
        # Save Circle Map
        fig = plt.figure(figsize = (15,15))
        plt.title(map_name+' Circle Map',fontsize=36)
        fig = sns.heatmap(c_circle_map,square=True,yticklabels=False,xticklabels=False,center = 0)
        fig.figure.savefig(graph_path+r'\\'+map_name+'_Circle_Map.png')
        plt.close()
        # Save Size Map
        fig = plt.figure(figsize = (15,15))
        plt.title(map_name+' Size Map',fontsize=36)
        fig = sns.heatmap(c_size_map,square=True,yticklabels=False,xticklabels=False,center = 0,cmap = 'coolwarm')
        fig.figure.savefig(graph_path+r'\\'+map_name+'_Size_Map.png')
        plt.close()
        # Save Stacked Circle Map
        folded_map = cv2.cvtColor(global_average,cv2.COLOR_GRAY2RGB)*0.7/256
        normed_c_circle_map = c_circle_map/abs(c_circle_map).max()
        posi_part = normed_c_circle_map*(normed_c_circle_map>0)
        nega_part = -normed_c_circle_map*(normed_c_circle_map<0)
        folded_map[:,:,2] += posi_part*255
        folded_map[:,:,0] += nega_part*255
        folded_map = np.clip(folded_map,0,255).astype('u1')
        gt.Show_Graph(folded_map, map_name+'_Folded', graph_path,0)
    # After map generation, save cell response data.
    ot.Save_Variable(graph_path, 'All_Map_Response', all_cell_response)
    
    return True

        







#%%
if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    day_folder = r'D:\Test_Data\2P\220421_L85'
    One_Key_T_Map(day_folder,'Run007',run_type = 'OD_2P')
    One_Key_T_Map(day_folder,'Run008',run_type = 'G16_2P')
    One_Key_T_Map(day_folder,'Run009',run_type = 'HueNOrien4')
# =============================================================================
#     #change mask file in celldics.
#     compare_dic = ot.Load_Variable(r'D:\Test_Data\2P\220421_L85\_CAIMAN\comp_id_dic.pkl')
#     acn = list(compare_dic.keys())
#     est = cnm2_file.estimates
#     for i,cc in enumerate(acn):
#         c_mask_new = np.reshape(est.A[:,compare_dic[cc]].toarray(), (512,512), order='F')
#         cell_dics[cc]['Cell_Mask'] = c_mask_new
#     ot.Save_Variable(r'D:\Test_Data\2P\220421_L85\_CAIMAN', 'All_Series_Dic.pkl', cell_dics)
# =============================================================================
    # Compare Cai 126(273) and raw 101 in Run07
    #frame 24016-25621