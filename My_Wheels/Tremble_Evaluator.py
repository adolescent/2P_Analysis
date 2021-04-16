# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 10:54:45 2020

@author: ZR
This file used to Evaluate align quality. Through mass center 

"""
from My_Wheels.Graph_Cutter import Graph_Cutter
import My_Wheels.OS_Tools_Kit as OS_Tools
import My_Wheels.Graph_Operation_Kit as Graph_Tools
import numpy as np
import cv2
from skimage import filters
from skimage.measure import regionprops
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import seaborn as sns

#%% Function1, Core function, calculate tremble in single folder.
def Tremble_Evaluator(
        data_folder,
        ftype = '.tif',
        boulder_ignore = 20,
        cut_shape = (9,9),
        mask_thres = 0
        ):
    all_file_name = OS_Tools.Get_File_Name(data_folder,ftype)
    template = cv2.imread(all_file_name[0],-1)
    origin_dtype = template.dtype
    graph_shape = template.shape
    graph_num = len(all_file_name)
    origin_graph_matrix = np.zeros(shape = graph_shape+(graph_num,),dtype = origin_dtype)
    for i in range(graph_num):
        origin_graph_matrix[:,:,i] = cv2.imread(all_file_name[i],-1)
    average_graph = origin_graph_matrix.mean(axis = 2).astype('u2')
    # Show schematic of cutted graph.
    schematic,_,_,_ = Graph_Cutter(average_graph,boulder_ignore,cut_shape)
    # Then,save cutted graphs into dics.
    cutted_graph_dic = {}
    fracture_num = cut_shape[0]*cut_shape[1]
    for i in range(fracture_num):# initialize cut dics.
        cutted_graph_dic[i]=[]
    for i in range(graph_num):# Cycle all graphs
        current_graph = origin_graph_matrix[:,:,i]
        _,_,_,cutted_graphs = Graph_Cutter(current_graph,boulder_ignore,cut_shape)
        for j in range(fracture_num):# save each fracture
            cutted_graph_dic[j].append(cutted_graphs[j])
    # Calculate graph center of each fracture trains. Use weighted center.
    all_frac_center = np.zeros(shape = (fracture_num,graph_num,2),dtype = 'f8')
    for i in range(fracture_num):
        current_frac = cutted_graph_dic[i]
        for j in range(graph_num):
            current_graph = current_frac[j]
            if mask_thres == 'otsu':
                thres = filters.threshold_otsu(current_graph)
            elif (type(mask_thres) == int or type(mask_thres) == float):
                thres = mask_thres
            else:
                raise IOError('Invalid mask threshold.')
            mask = (current_graph > thres).astype(int)
            properties = regionprops(mask, current_graph)
            current_mc = properties[0].weighted_centroid
            all_frac_center[i,j,:] = current_mc #In sequence YX
    return schematic,all_frac_center
    # Then, calculate final mass center location & graph ploting.
# =============================================================================
#     This part is example of subplot plotter. 
#     fig, ax = plt.subplots(3,3,figsize = (20,20))
#     ax[1,2].plot([1, 2, 3, 4], [1, 4, 9, 16])
#     t2 = np.arange(0, 5, 0.02)
#     ax[0,1].plot(t2, np.cos(2 * np.pi * t2), 'r--')
#     fig.savefig('FileName')
# =============================================================================
#%% Function 2, tremble compare and evaluation
def Tremble_Comparision(before_folder,
                        after_folder,
                        boulder_ignore = 20,
                        cut_shape = (9,9),
                        mask_thres = 0):
    # Initialization
    save_folder = after_folder+r'\Results'
    OS_Tools.mkdir(save_folder)
    save_folder = save_folder+r'\Tremble_Compare'
    OS_Tools.mkdir(save_folder)
    row_num = cut_shape[0]
    col_num = cut_shape[1]
    frac_num = row_num*col_num
    cov_matrix_dic = {}
    var_matrix_dic = {}
    variation = np.zeros(shape = (row_num,col_num,2),dtype = 'f8')
    variation_change = np.zeros(shape = (row_num,col_num),dtype = 'f8')
    variation_prop = np.zeros(shape = (row_num,col_num),dtype = 'f8')
    # Calculation Begins
    before_schematic,before_frac_center = Tremble_Evaluator(before_folder,boulder_ignore = boulder_ignore,cut_shape = cut_shape,mask_thres=mask_thres)
    after_schematic,after_frac_center = Tremble_Evaluator(after_folder,boulder_ignore = boulder_ignore,cut_shape = cut_shape,mask_thres=mask_thres)
    fig,ax = plt.subplots(row_num,col_num,figsize = (30,28))# Initialize graphs
    fig.suptitle('Mass Center Distribution', fontsize=54)
    # Cycle all fracture,get scatter map and variance 
    for i in range(frac_num):
        # Graph_Plot
        current_row = i%row_num
        current_col = i//row_num
        ax[current_row,current_col].scatter(before_frac_center[i,:,1],before_frac_center[i,:,0],s = 1,c = 'r')
        ax[current_row,current_col].scatter(after_frac_center[i,:,1],after_frac_center[i,:,0],s = 1,c = 'g')
        # After plot, calculate cov matrix and variance.
        before_cov = np.cov(before_frac_center[i,:,:].T)
        after_cov = np.cov(after_frac_center[i,:,:].T)
        cov_matrix_dic[i] = (before_cov,after_cov)
        before_eig,_ = np.linalg.eig(before_cov)
        after_eig,_ = np.linalg.eig(after_cov)
        before_var = np.round(before_eig.sum(),4)
        after_var = np.round(after_eig.sum(),4)
        variation[current_row,current_col,0] = before_var
        variation[current_row,current_col,1] = after_var
        variation_change[current_row,current_col] = before_var-after_var
        variation_prop[current_row,current_col] = (before_var-after_var)/before_var
        # Text annotate
        anchored_text = AnchoredText('Before variance:'+str(before_var)+'\n After variance:'+str(after_var), loc='lower left')
        ax[current_row,current_col].add_artist(anchored_text)
    
    # After this, save figures and matrixs.
    var_matrix_dic['Before'] = variation[:,:,0]
    var_matrix_dic['After'] = variation[:,:,1]
    Graph_Tools.Show_Graph(before_schematic, 'Before_Schematic', save_folder)
    Graph_Tools.Show_Graph(after_schematic, 'After_Schematic', save_folder)
    fig.savefig(save_folder+'\Scatter Plots.png',dpi = 330)
    OS_Tools.Save_Variable(save_folder, 'cov_matrix', cov_matrix_dic)
    OS_Tools.Save_Variable(save_folder, 'variance_matrix', var_matrix_dic)
    # Calculate variance change and plot variance map.
    # Before variance map
    plt.clf()
    fig2 = plt.figure(figsize = (15,15))
    plt.title('Before Align Variance',fontsize=36)
    fig2 = sns.heatmap(variation[:,:,0],cmap = 'bwr',annot=True,annot_kws={"size": 20},square=True,yticklabels=False,xticklabels=False,center = 0)
    fig2.figure.savefig(save_folder+'\Before_Variance.png',dpi = 330)
    # After variance map
    plt.clf()
    fig2 = plt.figure(figsize = (15,15))
    plt.title('After Align Variance',fontsize=36)
    fig2 = sns.heatmap(variation[:,:,1],cmap = 'bwr',annot=True,annot_kws={"size": 20},square=True,yticklabels=False,xticklabels=False,center = 0)
    fig2.figure.savefig(save_folder+'\After_Variance.png',dpi = 330)
    # Variance change map
    plt.clf()
    fig2 = plt.figure(figsize = (15,15))
    plt.title('Variance Change',fontsize=36)
    fig2 = sns.heatmap(variation_change,cmap = 'bwr',annot=True,annot_kws={"size": 20},square=True,yticklabels=False,xticklabels=False,center = 0)
    fig2.figure.savefig(save_folder+'\Variance_Change.png',dpi = 330)
    # Variance change propotion map
    plt.clf()
    fig2 = plt.figure(figsize = (15,15))
    plt.title('Variance Change Propotion',fontsize=36)
    fig2 = sns.heatmap(variation_prop,cmap = 'bwr',annot=True,annot_kws={"size": 20},square=True,yticklabels=False,xticklabels=False,center = 0)
    fig2.figure.savefig(save_folder+'\Variance_Change_Prop.png',dpi = 330)
    return cov_matrix_dic,var_matrix_dic
#%% Function3, leat tremble average graphs
def Least_Tremble_Average_Graph(
        data_folder,
        average_prop = 0.1,
        cut_shape = (9,9)
        ):
    all_tif_name = np.array(OS_Tools.Get_File_Name(data_folder))
    _,frac_disps = Tremble_Evaluator(data_folder,cut_shape = cut_shape)
    frac_num,frame_num,_ = frac_disps.shape
    # Then calculate average center and least error graph.
    frac_centers = np.zeros(shape = (frac_num,2),dtype = 'f8')
    for i in range(frac_num):
        frac_centers[i,0] = frac_disps[i,:,0].mean()
        frac_centers[i,1]  = frac_disps[i,:,1].mean()
    # And all frac_total movings
    total_movings = np.zeros(frame_num,dtype = 'f8')
    for i in range(frame_num):
        c_dist = 0
        for j in range(frac_num):
            c_dist += (frac_centers[j][0]-frac_disps[j,i,0])**2+(frac_centers[j][1]-frac_disps[j,i,1])**2
        total_movings[i] = c_dist
    # Then find least props.
    used_num = int(frame_num*average_prop)
    if used_num < 300:# least num of average is set to 300 to avoid problem.
        used_num = min(300,frame_num)
    print('Average of most stable '+str(used_num)+' Frames.')
    if used_num<300:# meaning all frame used
        graph_names = all_tif_name
    else:
        used_frame_ind = np.argpartition(total_movings,used_num)[0:used_num]
        graph_names = all_tif_name[used_frame_ind]
    averaged_graph = Graph_Tools.Average_From_File(graph_names)
    
    return averaged_graph,graph_names