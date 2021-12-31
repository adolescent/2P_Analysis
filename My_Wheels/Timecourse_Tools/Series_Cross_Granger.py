# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 13:34:07 2021

@author: ZR

"""
from statsmodels.tsa.stattools import grangercausalitytests as gctest
import pandas as pd
import numpy as np


def Granger_Core(series_A,series_B,lag = 5,sig_thres = 0.001):
    '''
    Granger test whether B cause A.

    Parameters
    ----------
    series_A : array_like
        Result series.
    series_B : array_like
        Reason series.
    lag : int, optional
        Max lag of granger test. The default is 5.
    sig_thres : float, optional
        Threshold of granger test. The default is 0.001.

    Returns
    -------
    sig_flag : (bool)
        Whether this result is significant.
    regression_model : (OLSResults)
        Result of granger fitted model.
    '''
    test_frame = pd.DataFrame([series_A,series_B]).T
    raw_gc_result = gctest(test_frame,maxlag = [lag],verbose = 0)
    raw_gc_result = list(raw_gc_result.values())[0]
    sig_flag = (sig_thres>raw_gc_result[0]['params_ftest'][1])
    regression_model = raw_gc_result[1][1]
    
    return sig_flag,regression_model


def Multi_components_Granger(input_frame,lag = 5,sig_thres = 0.001):
    '''
    Do multi component cross granger test.

    Parameters
    ----------
    input_frame : pd Frame
        Input time series. Each row shall be a series, columns as different times.
    lag : int, optional
        Lags used for granger test. The default is 5.
    sig_thres : float, optional
        Significant threshold for granger test. The default is 0.001.

    Returns
    -------
    GCA_Matrix : (pd Frame)
        Matrix of GCA Test Result. Only T/F given.

    '''
    all_PC_name = input_frame.index
    GCA_Matrix = pd.DataFrame(index = all_PC_name,columns = all_PC_name)
    for i,result_PC in enumerate(all_PC_name):
        result_series = input_frame.loc[result_PC,:]
        for j,reason_PC in enumerate(all_PC_name):
            reason_series = input_frame.loc[reason_PC,:]
            sig_flag,_ = Granger_Core(result_series, reason_series,lag = lag,sig_thres = sig_thres)
            GCA_Matrix.loc[result_PC,reason_PC] = sig_flag
    return GCA_Matrix
