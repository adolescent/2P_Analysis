# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 14:58:57 2022

@author: ZR
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm


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




def One_Key_Pairwise_Corr(day_folder,runname = 'Run001',start_frame = 0,
                          winstep = 60,winsize = 300):
    '''
    Generate window slide pairwise corr in onekey.

    Parameters
    ----------
    day_folder : TYPE
        DESCRIPTION.
    winstep : TYPE, optional
        DESCRIPTION. The default is 60.
    winsize : TYPE, optional
        DESCRIPTION. The default is 300.

    Returns
    -------
    None.

    '''
    pass