# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 10:43:23 2020

@author: ZR
"""

import os
import My_Wheels.OS_Tools_Kit as OS_Tools
from My_Wheels.Standard_Parameters.Sub_Graph_Dics import Sub_Dic_Generator
from My_Wheels.Translation_Align_Function import Translation_Alignment

def Standard_Stim_Processor(
             data_folder,
             stim_folder,
             run_name,
             tuning_graph = True,
             cell_method = 'Default'
             
             ):
    '''
    Input part of this module.

    Parameters
    ----------
    graph_folder : (str)
        Input graph folder.
    stim_folder : (str)
        Folder of correlated stimulus.
    run_name : ('OD_2P','OD_OI','G8','RGLum4')
        Run name. key as graph name, value are list.
    tuning_graph : (bool), optional
        Whether we produce . The default is True.
    cell_method: ('Default' or cell file path)
        If default, use on-off graph, else you need to give the input path.
    
    Returns
    -------
    None.

    '''
    raw_tif_name = OS_Tools.Get_File_Name(data_folder)    
    work_folder = data_folder+r'\Results'
    OS_Tools.mkdir(work_folder)
    aligned_frame_folder = work_folder+r'\Aligned_Frames'
    OS_Tools.mkdir(aligned_frame_folder)
    # Step1, align graphs. If already aligned, just read 
    if not os.listdir(aligned_frame_folder): # if this is a new folder
        Translation_Alignment([data_folder])
        aligned_all_tif_name = OS_Tools.Get_File_Name(aligned_frame_folder)
    else:# If folder is not empty, just read aligned graphs.
        aligned_all_tif_name = OS_Tools.Get_File_Name(aligned_frame_folder)
        
