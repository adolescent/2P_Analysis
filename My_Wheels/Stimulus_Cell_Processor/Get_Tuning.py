# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 11:50:23 2022

@author: ZR
"""

import OS_Tools_Kit as ot


def Get_Tuned_Cells(day_folder,tuning_name,thres = 0.05):
    '''
    
    Get tuned cells from tuning dics

    Parameters
    ----------
    day_folder : (str)
        Folder of one day run.
    tuning_name : (str)
        Name of tuning you want to find(e.g. 'LE'). Must be in .tuning files.
    thres : (float), optional
        Threshold of p significant. The default is 0.05.

    Returns
    -------
    tuned_cells : (list)
        List of cell have tuning above.

    '''
    tuning_dic = ot.Load_Variable(ot.Get_File_Name(day_folder,'.tuning')[0])
    acn = list(tuning_dic.keys())
    tuned_cells = []
    for i,cc in enumerate(acn):
        c_dic = tuning_dic[cc]
        if tuning_name in c_dic:
            if c_dic[tuning_name]['t_value']>0 and c_dic[tuning_name]['p_value']<thres:
                tuned_cells.append(cc)
    return tuned_cells