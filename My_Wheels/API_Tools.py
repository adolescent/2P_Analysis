# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 11:01:35 2021

@author: ZR

These functions are speciallized for function connection.
As thought might change by time, API need to be adjusted by time.
"""
import pandas as pd
import OS_Tools_Kit as ot


#%% Function1, All cell data to pandas frame
def All_Cell_To_PD(day_folder,runname,mode = 'raw'):
    '''
    Get Single run data frame from .ac file.

    Parameters
    ----------
    day_folder : (str)
        Day folder of data. Cell data need to be generated first.
    runname : (str)
        Data of which run? e.g. 'Run001'
    mode : 'raw' or 'processed', optional
        Use F train or dF/F train. The default is 'raw'.

    Returns
    -------
    cell_frame : (pd Frame)
        Pandas frame of cell data, rows are cells, columns are each frames.

    '''
    ac_fn = ot.Get_File_Name(day_folder,'.ac')[0]
    ac_dic = ot.Load_Variable(ac_fn)
    acn = list(ac_dic.keys())
    cell_frame = pd.DataFrame()
    for i in range(len(acn)):
        c_cn = acn[i]
        tc = ac_dic[c_cn]
        if mode == 'raw':
            tc_train = tc[runname]['F_train']
        elif mode == 'processed':
            tc_train = tc[runname]['dF_F_train']
        cell_frame[c_cn] = tc_train
    # To make column as dimension, row as cell num, we need a transfer.
    cell_frame = cell_frame.T
    return cell_frame

