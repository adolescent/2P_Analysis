# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 16:46:51 2021

@author: ZR
"""
import OS_Tools_Kit as ot
import numpy as np
import pandas as pd
from Filters import Signal_Filter


def Pre_Processor(day_folder,runname = 'Run001',
                  start_time = 0,stop_time = 99999,
                  fps = 1.301,passed_band = (0.05,0.5),order = 5):
    '''
    Proprocess of spontaneous data frame.

    Parameters
    ----------
    day_folder : (str)
        Day folder of 2p datas.
    runname : (str), optional
        Which run? The default is 'Run001'.
    start_time : (int), optional
        Time of series start (in second). The default is 0.
    stop_time : (int), optional
        Time of series end(in second). If exceed use last frame. The default is 99999.
    fps : (float), optional
        Capture frequency. The default is 1.301.
    passed_band : (2-element-turple), optional
        High pass and Low pass frequency. The default is (0.05,0.5).

    Returns
    -------
    processed_cell_frame : (pd Frame)
        Cell data frame after filter & cut.

    '''
    
    cell_file_name = ot.Get_File_Name(day_folder,'.ac')[0]
    cell_dic = ot.Load_Variable(cell_file_name)
    acn = list(cell_dic.keys())
    raw_frame = pd.DataFrame(columns = acn)
    start_frame = int(start_time*fps)
    
    for i,ccn in enumerate(acn):
        cc = cell_dic[ccn]
        if cc['In_Run'][runname]:
            c_series = cc[runname]['F_train']
            filted_c_series = Signal_Filter(c_series,order,filter_para = (passed_band[0]*2/fps,passed_band[1]*2/fps))
            # then calculate dF/F series.
            stop_frame = int(min(stop_time*fps,len(filted_c_series)))
            used_filted_c_series = filted_c_series[start_frame:stop_frame]
            c_dF_F_series = (used_filted_c_series-used_filted_c_series.mean())/used_filted_c_series.mean()
            raw_frame.loc[:,ccn] = c_dF_F_series
    processed_cell_frame = raw_frame.T
    processed_cell_frame = processed_cell_frame.dropna().copy()# to avoid highly fragment warning
    return processed_cell_frame
            
def Pre_Processor_By_Frame(input_frame,fps = 1.301,passed_band=(0.05,0.5),order = 5):
    all_cell_name = input_frame.index.tolist()
    raw_frame = pd.DataFrame(columns = all_cell_name)
    for i,ccn in enumerate(all_cell_name):
        c_series = np.array(input_frame.loc[ccn,:])
        filted_c_series = Signal_Filter(c_series,order,filter_para = (passed_band[0]*2/fps,passed_band[1]*2/fps))
        # then calculate dF/F series.
        c_dF_F_series = (filted_c_series-filted_c_series.mean())/filted_c_series.mean()
        raw_frame.loc[:,ccn] = c_dF_F_series
    processed_cell_frame = raw_frame.T
    processed_cell_frame = processed_cell_frame.copy()
    return processed_cell_frame
            
def Pre_Processor_Multi_Run(day_folder,run_list,start_time = 0,stop_time = 99999,
                            fps = 1.301,passed_band = (0.05,0.5),order = 5):
    
    
    cell_file_name = ot.Get_File_Name(day_folder,'.ac')[0]
    cell_dic = ot.Load_Variable(cell_file_name)
    acn = list(cell_dic.keys())
    start_frame = int(start_time*fps)
    # Select used cell here.
    used_cell_dic = {}
    for i,ccn in enumerate(acn):
        cc = cell_dic[ccn]
        flag_use = True
        for j,c_run in enumerate(run_list):
            flag_use *= cc['In_Run'][c_run]
        if flag_use == True:
            used_cell_dic[ccn] = cc
    del cell_dic,acn
    used_acn = list(used_cell_dic.keys())
    raw_frame = pd.DataFrame(columns = used_acn)
    for i,ccn in enumerate(used_acn):
        cc = used_cell_dic[ccn]
        c_series = np.array([])
        for j,c_run in enumerate(run_list):
            c_series = np.append(c_series,cc[c_run]['F_train'])
        filted_c_series = Signal_Filter(c_series,order,filter_para = (passed_band[0]*2/fps,passed_band[1]*2/fps))
        # then calculate dF/F series.
        stop_frame = int(min(stop_time*fps,len(filted_c_series)))
        used_filted_c_series = filted_c_series[start_frame:stop_frame]
        c_dF_F_series = (used_filted_c_series-used_filted_c_series.mean())/used_filted_c_series.mean()
        raw_frame.loc[:,ccn] = c_dF_F_series
    processed_cell_frame = raw_frame.T
    processed_cell_frame = processed_cell_frame.dropna().copy()# to avoid highly fragment warning
    return processed_cell_frame