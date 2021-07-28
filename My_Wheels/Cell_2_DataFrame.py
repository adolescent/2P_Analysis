# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 11:01:35 2021

@author: ZR

These functions are speciallized for function connection.
As thought might change by time, API need to be adjusted by time.
"""
import pandas as pd
import OS_Tools_Kit as ot
import numpy as np

#%% Function1, Single Run to Data Frame.
def Single_Run_Fvalue_Frame(day_folder,runname):
    '''
    Get Single run data frame from .ac file. ONLY F 

    Parameters
    ----------
    day_folder : (str)
        Day folder of data. Cell data need to be generated first.
    runname : (str)
        Data of which run? e.g. 'Run001'

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
        if tc['In_Run'][runname]:
            tc_train = tc[runname]['F_train']
            cell_frame[c_cn] = tc_train
    # To make column as dimension, row as cell num, we need a transfer.
    cell_frame = cell_frame.T
    return cell_frame

#%% Function2, Multiple Run Cat
def Multi_Run_Fvalue_Cat(day_folder,runlists,rest_time = (600,600),fps = 1.301):
    
    ac_fn = ot.Get_File_Name(day_folder,'.ac')[0]
    ac_dic = ot.Load_Variable(ac_fn)
    acn = list(ac_dic.keys())
    All_Data_Frame = pd.DataFrame()
    # Get all runs one by one.
    for i in range(len(runlists)):
        c_runname = runlists[i]
        c_run_frame = pd.DataFrame()
        for j in range(len(acn)):
            c_cn = acn[j]
            tc = ac_dic[c_cn]
            if tc['In_Run'][c_runname]:
                tc_train = tc[c_runname]['F_train']
                # add blank frame as last value.
                if i <(len(runlists)-1):
                    blank_frames = int(rest_time[i]*fps)
                    blank_array = np.ones(blank_frames)*tc_train[-1]
                    tc_train = np.concatenate((tc_train,blank_array), axis=0)
                c_run_frame[c_cn] = tc_train
        # To make column as dimension, row as cell num, we need a transfer.
        c_run_frame = c_run_frame.T
        All_Data_Frame = pd.concat([All_Data_Frame,c_run_frame],axis = 1)
    # Remove any zero frame, avoid null cell.
    All_Data_Frame = All_Data_Frame.dropna(axis =0,how ='any')
    return All_Data_Frame
    
    
    

