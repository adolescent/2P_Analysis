# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 12:38:13 2021

@author: ZR
"""
import numpy as np
import pandas as pd
from scipy import stats
from Decorators import Timer

@Timer
def Pair_Corr_Core(data_frame,cell_names,set_name = 'All_Cells',method = 'spearman'):
    
    '''
    Core function of all pair corr. Series shall be cutted previously.

    Parameters
    ----------
    data_frame : (pd Frame)
        Cell series with name.
    cell_names : (list)
        List of cell you want to do corr.
    set_name : (str),optional
        Name of cell sets. The default is 'All_Cells'

    Returns
    -------
    pair_corr : (pd Frame)
        Correlation data frame with corr information.Single column, but normalized.

    '''
    cell_num = len(cell_names)
    pair_corr = pd.DataFrame(index=[set_name])
    for i,A_cell in enumerate(cell_names):
        A_series = np.array(data_frame.loc[A_cell])
        for j in range(i+1,cell_num):
            B_cell = cell_names[j]
            c_pair_name = A_cell[-4:]+'*'+B_cell[-4:]
            B_series = np.array(data_frame.loc[B_cell])
            if method == 'spearman':
                c_corr = (stats.spearmanr(A_series,B_series)[0])
            elif method == 'pearson':
                c_corr = (stats.pearsonr(A_series,B_series)[0])
            pair_corr[c_pair_name] = c_corr
    pair_corr = pair_corr.T
    return pair_corr

@Timer
def Pair_Corr_Window_Slide(data_frame,cell_names,window_size = 300,window_step = 60,fps = 1.301,method = 'spearman'):
    '''
    Calculate window slide pairwise correlation. VERY SLOW...

    Parameters
    ----------
    data_frame : (pd Frame)
        Cell data trains.
    cell_names : (list)
        List of cells you want to do pairwise correlation.
    window_size : (int), optional
        Size of correlation window (in seconds). The default is 300.
    window_step : (int), optional
        Step of correlation window (in seconds). The default is 60.
    fps : (float), optional
        Capture frequency (Hz). The default is 1.301.
    method : ('spearman' or 'pearson'), optional
        Correlation method. The default is 'spearman'.

    Returns
    -------
    pair_corr_frames : (pd Frame)
        Data Frame of .

    '''
    cell_num = len(cell_names)
    win_frame_size = int(window_size*fps)
    win_frame_step = int(window_step*fps)
    window_num = 1+int((data_frame.shape[1]-win_frame_size)/win_frame_step)
    pair_corr_frames = pd.DataFrame(index = range(window_num))
    for i in range(window_num):
        c_window = data_frame.iloc[:,i*win_frame_step:(i*win_frame_step+win_frame_size)]
        for j,A_cell in enumerate(cell_names):
            A_series = np.array(c_window.loc[A_cell])
            for k in range(j+1,cell_num):
                B_cell = cell_names[k]
                c_pair_name = A_cell[-4:]+'*'+B_cell[-4:]
                B_series = np.array(c_window.loc[B_cell])
                if method == 'spearman':
                    c_corr = (stats.spearmanr(A_series,B_series)[0])
                elif method == 'pearson':
                    c_corr = (stats.pearsonr(A_series,B_series)[0])
                pair_corr_frames.loc[i:i,c_pair_name] = c_corr
    pair_corr_frames = pair_corr_frames.T
    return pair_corr_frames



@Timer
def Sort_Corr_By_Mean(pair_corr_frames,mean_range = (0,99999)):
    pass
    return sorted_corr_frames



def Corr_Histo(pari_corr_frames,bins,corr_lim = (0,1)):
    return histo_frames


