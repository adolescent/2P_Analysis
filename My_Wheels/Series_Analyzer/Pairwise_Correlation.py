# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 14:58:57 2022

@author: ZR
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm
from Series_Analyzer.Series_Cutter import Series_Window_Slide


def Pairwise_Corr_Core(all_cell_dic,tuning_dic,data_frame):
    '''
    Core corr function, generate pairwise corr of given data frame.

    Parameters
    ----------
    all_cell_dic : (dic)
        All cell dic. Including all cell informations.
    tuning_dic : (dic)
        Tuning dic, usually 'Cell_Tuning_Dic' file.
    data_frame : (pd Frame)
        Cell response frames.

    Returns
    -------
    Pair_Corr : (pd Frame)
        Pairwise correlation data pairs.

    '''
    acn = list(all_cell_dic.keys())
    cell_num = len(acn)
    pair_num = int(cell_num*(cell_num-1)/2)
    Pair_Corr = pd.DataFrame(columns = ['Corr','p','CellA','CellB','Dist','OD_A','OD_B','Orien_A','Orien_B'],index = range(pair_num))
    counter = 0
    
    for i in tqdm(range(len(acn))):
        cell_A = acn[i]
        cell_A_series = data_frame.loc[cell_A,:]
        cell_A_loc = all_cell_dic[cell_A]['Cell_Loc']
        od_A = tuning_dic[cell_A]['OD']['Tuning_Index']
        orien_A = tuning_dic[cell_A]['Fitted_Orien']
        for j in range(i+1,len(acn)):
            cell_B = acn[j]
            cell_B_series = data_frame.loc[cell_B,:]
            cell_B_loc = all_cell_dic[cell_B]['Cell_Loc']
            od_B = tuning_dic[cell_B]['OD']['Tuning_Index']
            orien_B = tuning_dic[cell_B]['Fitted_Orien']
            # calculate corr and dist.
            dist = np.linalg.norm(cell_A_loc-cell_B_loc)
            corr,p = pearsonr(cell_A_series,cell_B_series)
            Pair_Corr.loc[counter] = [corr,p,cell_A,cell_B,dist,od_A,od_B,orien_A,orien_B]
            counter +=1
    Pair_Corr = Pair_Corr.replace('No_Tuning',-999)
    Pair_Corr['Corr'] = Pair_Corr['Corr'].astype('f8')
    Pair_Corr['p'] = Pair_Corr['p'].astype('f8')
    Pair_Corr['Dist'] = Pair_Corr['Dist'].astype('f8')
    Pair_Corr['OD_A'] = Pair_Corr['OD_A'].astype('f8')
    Pair_Corr['OD_B'] = Pair_Corr['OD_B'].astype('f8')
    Pair_Corr['Orien_A'] = Pair_Corr['Orien_A'].astype('f8')
    Pair_Corr['Orien_B'] = Pair_Corr['Orien_B'].astype('f8')
    
    
    return Pair_Corr




def Series_Cut_Pair_Corr(all_cell_dic,tuning_dic,data_frame,win_size = 300,win_step = 60):
    
    cutted_frame = Series_Window_Slide(data_frame,win_size,win_step)
    frac_num = cutted_frame.shape[2]
    # generate info first.
    acn = list(all_cell_dic.keys())
    cell_num = len(acn)
    pair_num = int(cell_num*(cell_num-1)/2)
    Pair_Corr_info = pd.DataFrame(columns = ['CellA','CellB','Dist','OD_A','OD_B','Orien_A','Orien_B'],index = range(pair_num))
    Pair_Corr_windowed = pd.DataFrame(columns = range(frac_num),index = range(pair_num))
    counter = 0
    print('Generating corr info.. \n')
    for i in tqdm(range(len(acn))):
        cell_A = acn[i]
        cell_A_loc = all_cell_dic[cell_A]['Cell_Loc']
        od_A = tuning_dic[cell_A]['OD']['Tuning_Index']
        orien_A = tuning_dic[cell_A]['Fitted_Orien']
        for j in range(i+1,len(acn)):
            cell_B = acn[j]
            cell_B_loc = all_cell_dic[cell_B]['Cell_Loc']
            od_B = tuning_dic[cell_B]['OD']['Tuning_Index']
            orien_B = tuning_dic[cell_B]['Fitted_Orien']
            # calculate corr and dist.
            dist = np.linalg.norm(cell_A_loc-cell_B_loc)
            Pair_Corr_info.loc[counter] = [cell_A,cell_B,dist,od_A,od_B,orien_A,orien_B]
            
            # cycle time window here.
            for k in range(frac_num):
                c_A_series = cutted_frame[i,:,k]
                c_B_series = cutted_frame[j,:,k]
                corr,_ = pearsonr(c_A_series,c_B_series)
                Pair_Corr_windowed.loc[counter,k] = corr
                
            counter +=1
    Pair_Corr_info = Pair_Corr_info.replace('No_Tuning',-999)
    Pair_Corr_info['Dist'] = Pair_Corr_info['Dist'].astype('f8')
    Pair_Corr_info['OD_A'] = Pair_Corr_info['OD_A'].astype('f8')
    Pair_Corr_info['OD_B'] = Pair_Corr_info['OD_B'].astype('f8')
    Pair_Corr_info['Orien_A'] = Pair_Corr_info['Orien_A'].astype('f8')
    Pair_Corr_info['Orien_B'] = Pair_Corr_info['Orien_B'].astype('f8')

    for i in range(frac_num):
        Pair_Corr_windowed[i] = Pair_Corr_windowed[i].astype('f8')

    
    
    return Pair_Corr_info,Pair_Corr_windowed