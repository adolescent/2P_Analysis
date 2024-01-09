'''
This script give basic infos of all used location on all data points.
Including repeat frequency and cell basic infos.

'''
#%%
from Cell_Class.Stim_Calculators import Stim_Cells
from Cell_Class.Format_Cell import Cell
import OS_Tools_Kit as ot
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import cv2
from Kill_Cache import kill_all_cache
from sklearn.model_selection import cross_val_score
from sklearn import svm
import umap
import umap.plot
from scipy.stats import pearsonr
import scipy.stats as stats
from Cell_Class.Plot_Tools import Plot_3D_With_Labels
import copy
from Filters import Signal_Filter
from Cell_Class.Advanced_Tools import *
from Cell_Class.UMAP_Classifier_Analyzer import *


work_path = r'D:\_Path_For_Figs\_2312_ver2\Fig1'
all_path_dic = list(ot.Get_Sub_Folders(r'D:\_All_Spon_Data_V1'))
all_path_dic.pop(4)
all_path_dic.pop(6)

#%%################# GENERATE FRAME ################## 
basic_frame = pd.DataFrame(columns = ['Loc','Cell_Num','LE_Num','RE_Num','Orien0_Num','Orien45_Num','Orien90_Num','Orien135_Num','Red_Num','Green_Num','Blue_Num','Spon_len','Spon_prop','OD_prop','Orien_prop','Color_prop'])
thres = 0.01
for i,c_loc in tqdm(enumerate(all_path_dic)):
    c_loc_name = c_loc.split('\\')[-1]
    ac = ot.Load_Variable_v2(c_loc,'Cell_Class.pkl')
    c_spon = ot.Load_Variable(c_loc,'Spon_Before.pkl')
    c_model = ot.Load_Variable(c_loc,'All_Stim_UMAP_3D_20comp.pkl')
    ## part 1, cell tuning count parts
    cell_num = len(ac.acn)
    le_num = 0
    re_num = 0
    o0_num = 0
    o45_num = 0
    o90_num = 0
    o135_num = 0
    red_num = 0
    green_num = 0
    blue_num = 0
    od_map = ac.OD_t_graphs['OD']
    hv_map = ac.Orien_t_graphs['H-V']
    ao_map = ac.Orien_t_graphs['A-O']
    red_map = ac.Color_t_graphs['Red-White']
    green_map = ac.Color_t_graphs['Green-White']
    blue_map = ac.Color_t_graphs['Blue-White']    
    for j,cc in enumerate(ac.acn):
        c_od_response = od_map[cc]
        if c_od_response['t_value']>0 and c_od_response['p_value']<thres:
            le_num+=1
        elif c_od_response['t_value']<0 and c_od_response['p_value']<thres:
            re_num+=1
        c_hv_response = hv_map[cc]
        if c_hv_response['t_value']>0 and c_hv_response['p_value']<thres:
            o0_num +=1 
        elif c_hv_response['t_value']<0 and c_hv_response['p_value']<thres:
            o90_num +=1
        c_ao_response = ao_map[cc]
        if c_ao_response['t_value']>0 and c_ao_response['p_value']<thres:
            o45_num +=1 
        elif c_ao_response['t_value']<0 and c_ao_response['p_value']<thres:
            o135_num +=1
        c_r_response = red_map[cc]
        if c_r_response['t_value']>0 and c_r_response['p_value']<thres:
            red_num += 1
        c_g_response = green_map[cc]
        if c_g_response['t_value']>0 and c_g_response['p_value']<thres:
            green_num += 1
        c_b_response = blue_map[cc]
        if c_b_response['t_value']>0 and c_b_response['p_value']<thres:
            blue_num += 1

    ## part 2, spon freq parts.
    analyzer = UMAP_Analyzer(ac = ac,umap_model=c_model,spon_frame=c_spon,od = True,orien = True,color = True,isi = True)
    analyzer.Train_SVM_Classifier()
    spon_label = analyzer.spon_label
    spon_num = np.sum(spon_label>0)
    eye_num = np.sum((spon_label>0)*(spon_label<9))
    orien_num = np.sum((spon_label>8)*(spon_label<17))
    color_num = np.sum((spon_label>16)*(spon_label<23))
    spon_prop = spon_num/len(c_spon)
    od_prop = eye_num/len(c_spon)
    orien_prop = orien_num/len(c_spon)
    color_prop = color_num/len(c_spon)
    basic_frame.loc[len(basic_frame),:] = [c_loc_name,cell_num,le_num,re_num,o0_num,o45_num,o90_num,o135_num,red_num,green_num,blue_num,len(c_spon),spon_prop,od_prop,orien_prop,color_prop]
#%%######################DO THE SAME ON V2###########################
    
all_path_dic_v2 = list(ot.Get_Sub_Folders(r'D:\_All_Spon_Datas_V2'))
basic_frame = pd.DataFrame(columns = ['Loc','Cell_Num','LE_Num','RE_Num','Orien0_Num','Orien45_Num','Orien90_Num','Orien135_Num','Red_Num','Green_Num','Blue_Num','Spon_len','Spon_prop','OD_prop','Orien_prop','Color_prop'])
thres = 0.01
for i,c_loc in tqdm(enumerate(all_path_dic)):
    c_loc_name = c_loc.split('\\')[-1]
    ac = ot.Load_Variable_v2(c_loc,'Cell_Class.pkl')
    c_spon = ot.Load_Variable(c_loc,'Spon_Before.pkl')
    c_model = ot.Load_Variable(c_loc,'All_Stim_UMAP_3D_20comp.pkl')
    ## part 1, cell tuning count parts
    cell_num = len(ac.acn)
    le_num = 0
    re_num = 0
    o0_num = 0
    o45_num = 0
    o90_num = 0
    o135_num = 0
    red_num = 0
    green_num = 0
    blue_num = 0
    hv_map = ac.Orien_t_graphs['H-V']
    ao_map = ac.Orien_t_graphs['A-O']
    for j,cc in enumerate(ac.acn):
        c_hv_response = hv_map[cc]
        if c_hv_response['t_value']>0 and c_hv_response['p_value']<thres:
            o0_num +=1 
        elif c_hv_response['t_value']<0 and c_hv_response['p_value']<thres:
            o90_num +=1
        c_ao_response = ao_map[cc]
        if c_ao_response['t_value']>0 and c_ao_response['p_value']<thres:
            o45_num +=1 
        elif c_ao_response['t_value']<0 and c_ao_response['p_value']<thres:
            o135_num +=1
    analyzer = UMAP_Analyzer(ac = ac,umap_model=c_model,spon_frame=c_spon,od = False,orien = True,color = False,isi = True)
    analyzer.Train_SVM_Classifier()
    spon_label = analyzer.spon_label
    spon_num = np.sum(spon_label>0)
    eye_num = np.sum((spon_label>0)*(spon_label<9))
    orien_num = np.sum((spon_label>8)*(spon_label<17))
    color_num = np.sum((spon_label>16)*(spon_label<23))
    spon_prop = spon_num/len(c_spon)
    od_prop = eye_num/len(c_spon)
    orien_prop = orien_num/len(c_spon)
    color_prop = color_num/len(c_spon)
    basic_frame.loc[len(basic_frame),:] = [c_loc_name,cell_num,le_num,re_num,o0_num,o45_num,o90_num,o135_num,red_num,green_num,blue_num,len(c_spon),spon_prop,od_prop,orien_prop,color_prop]