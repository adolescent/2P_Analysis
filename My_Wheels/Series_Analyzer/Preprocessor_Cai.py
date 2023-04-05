# -*- coding: utf-8 -*-
"""
Created on Fri May 27 16:49:39 2022

@author: adolescent

Series preprocessor used on caiman data.
"""


import OS_Tools_Kit as ot
import numpy as np
import pandas as pd
from Filters import Signal_Filter
import List_Operation_Kit as lt

def Pre_Processor_Cai(day_folder,runname = 'Run001',subfolder = '_CAIMAN',
                  start_frame = 0,stop_frame = 99999,
                  fps = 1.301,passed_band = (0.005,0.3),order = 7,
                  base_mode = 'average',prop = 0.05,use_z = True):
    '''
    Proprocess of spontaneous data frame.

    Parameters
    ----------
    day_folder : (str)
        Day folder of 2p datas.
    runname : (str), optional
        Which run? The default is 'Run001'.
    start_frame : (int), optional
        Series start (in frame). The default is 0.
    stop_frame : (int), optional
        Series end(in frame). If exceed use last frame. The default is 99999.
    fps : (float), optional
        Capture frequency. The default is 1.301.
    passed_band : (2-element-turple), optional
        High pass and Low pass frequency. The default is (0.05,0.5).
        
        
    base_mode : ('average' or 'most_unactive'),optional
        Method of F0 selection. The default is 'most_unactive'
    prop : (float),optional
        Propotion of F0 for most unactive. The default is 0.05.

    Returns
    -------
    processed_cell_frame : (pd Frame)
        Cell data frame after filter & cut.

    '''
    runname = lt.Change_Runid_Style([runname])[0]
    cell_dic = ot.Load_Variable(ot.join(day_folder,subfolder),'All_Series_Dic.pkl')
    acn = list(cell_dic.keys())
    raw_frame = pd.DataFrame(columns = acn)
    for i,ccn in enumerate(acn):
        cc = cell_dic[ccn]
        c_series = cc[runname]
        filted_c_series = Signal_Filter(c_series,order,filter_para = (passed_band[0]*2/fps,passed_band[1]*2/fps))
        # then calculate dF/F series.
        used_filted_c_series = filted_c_series[start_frame:stop_frame]
        if base_mode == 'average':
            c_dF_F_series = (used_filted_c_series-used_filted_c_series.mean())/used_filted_c_series.mean()
            if use_z == True:
                c_dF_F_series = c_dF_F_series/c_dF_F_series.std()
        elif base_mode == 'most_unactive':
            base_num = int(len(used_filted_c_series)*prop)
            base_id = np.argpartition(used_filted_c_series, base_num)[:base_num]
            base = used_filted_c_series[base_id].mean()
            c_dF_F_series = (used_filted_c_series-base)/base
            if use_z == True:
                c_dF_F_series = c_dF_F_series/c_dF_F_series.std()
        else:
            raise IOError('Invalid F0 mode.')
        raw_frame.loc[:,ccn] = c_dF_F_series
    processed_cell_frame = raw_frame.T
    processed_cell_frame = processed_cell_frame.dropna().copy()# to avoid highly fragment warning
    return processed_cell_frame

    