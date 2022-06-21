# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 12:38:13 2021

@author: ZR
"""
import numpy as np
import pandas as pd
from scipy import stats
from Decorators import Timer
from Statistic_Tools import T_Test_Pair

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
    print('Function Not Finished Yet.')
    sorted_corr_frames = None
    return sorted_corr_frames



def Corr_Histo(pair_corr_frames,bins = 200,corr_lim = 'auto'):
    '''
    Generate histogram for Windows pair correlation graphs.

    Parameters
    ----------
    pair_corr_frames : (pd Frame)
        Windowed correlation data frames.
    bins : (int),optional
        How many bins you want to plot. The default is 200.
    corr_lim : (2-element-turple), optional
        Limitation of correlations. Set auto to skip. The default is 'auto'.

    Returns
    -------
    histo_frames : (pd Frame)
        .

    '''
    histo_frames = pd.DataFrame(columns = range(pair_corr_frames.shape[1]))
    # Get range from data
    example_slice = pair_corr_frames.iloc[:,0]
    if corr_lim == 'auto':
        bin_boulders = np.histogram(example_slice,bins=bins)[1] # This is 1 bit bigger than bin number(right boulder)
        corr_lim = (bin_boulders[0],bin_boulders[-1])
    else:
        bin_boulders = np.linspace(corr_lim[0],corr_lim[1],bins+1)
    # Cycle time windows
    for i in range(pair_corr_frames.shape[1]):
        c_window = pair_corr_frames.iloc[:,i]
        c_histo = np.histogram(c_window,bins = bins,range=corr_lim,density = True)[0]
        histo_frames[i] = c_histo
    # Rename row names
    row_dic = {}
    for i in range(len(bin_boulders)-1):
        row_dic[i] = round(bin_boulders[i],2)
    histo_frames = histo_frames.rename(index = row_dic)
    histo_frames = histo_frames.iloc[::-1]# Reverse y axis.
    # Calculate t value from first time window.
    t_train = []
    origin_disp = pair_corr_frames.iloc[:,0]
    for i in range(pair_corr_frames.shape[1]):
        target_disp = pair_corr_frames.iloc[:,i]
        c_tvalue,_,_ = T_Test_Pair(target_disp,origin_disp)
        t_train.append(c_tvalue)
        
    return histo_frames,t_train


def Window_by_Window_T_Testor(dataframe_A,dataframe_B,thres = 0.001):
    
    all_window = dataframe_A.columns.tolist()
    t_series = []
    p_series = []
    for i,c_win in enumerate(all_window):
        c_A_series = dataframe_A.iloc[:,i].tolist()
        c_B_series = dataframe_B.iloc[:,i].tolist()
        c_t,c_p,_ = T_Test_Pair(c_A_series, c_B_series)
        if c_p>thres:
            c_t = 0
        t_series.append(c_t)
        p_series.append(c_p)
    return t_series,p_series