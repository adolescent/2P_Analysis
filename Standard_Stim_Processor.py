# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 10:43:23 2020

@author: ZR
"""

import os
import My_Wheels.OS_Tools_Kit as OS_Tools
import My_Wheels.Graph_Operation_Kit as Graph_Tools

from My_Wheels.Translation_Align_Function import Translation_Alignment


def Standard_Stim_Processor(
             data_folder,
             stim_folder,
             sub_dic,
             tuning_graph = True,
             cell_method = 'Default',
             sub_method = 'dF/F',
             gaussian_parameter = ((5,5),1.5)
             ):
    '''
    Input part of this module. Althought this model can do align, if you want to align all graphs together, prealign is advised.

    Parameters
    ----------
    graph_folder : (str)
        Input graph folder. If graph has been aligned, this will use aligned graph directly. Else this function will align single run.
    stim_folder : (str)
        Folder of stimulus, or aligned stimulus file name. This will be checked later, if is folder, 
    sub_dic : (dic)
        Subtraction ID dictionary. Standard ID can be acquired from My_Wheels.Standard_Parameters.Sub_Graph_Dics. 
    tuning_graph : (bool), optional
        Whether we produce radar map. The default is True.
    cell_method: ('Default' or cell file path)
        If default, use on-off graph, else you need to give the input path.
    sub_method : ('dF' or 'dF/F')
        Method to generate subgraph. If dF,just subtraction; else we will /cond0 average. T test not affected. The default is 'dF/F'.
    gaussian_parameter : (turple)
        This step can be skipped if you input 'False'. Gaussian blur of graph process. Every graph will be filtered before process. The default is((5,5),1.5),meaning kernel of filter is (5,5), std = 1.5. 
    
    Returns
    -------
    None.

    '''
    # Path Cycle.
    work_folder = data_folder+r'\Results'
    OS_Tools.mkdir(work_folder)
    aligned_frame_folder = work_folder+r'\Aligned_Frames'
    OS_Tools.mkdir(aligned_frame_folder)
    
    # Step1, align graphs. If already aligned, just read 
    if not os.listdir(aligned_frame_folder): # if this is a new folder
        print('Aligned data not found. Aligning here..')
        Translation_Alignment([data_folder])
        aligned_all_tif_name = OS_Tools.Get_File_Name(aligned_frame_folder)
    else:# If folder is not empty, just read aligned graphs.
        aligned_all_tif_name = OS_Tools.Get_File_Name(aligned_frame_folder)
        
    # Step2, get stim fram align matrix. If already aligned, just read in aligned dictionary.
    file_detector = len(stim_folder.split('.'))
    if file_detector == 1:# Which means input is a folder
        print('Frame Stim not Aligned, aligning here...')
        from My_Wheels.Stim_Frame_Align import Stim_Frame_Align
        _,Frame_Stim_Dic = Stim_Frame_Align(stim_folder)
    else: # Input is a file
        OS_Tools.Load_Variable(stim_folder)
        
    # Step3, get cell information 
    if cell_method == 'Default':# meaning we will use On-Off graph to find cell.
        print('Cell information not found. Finding here..')
        average_graph = Graph_Tools.Average_From_File(aligned_all_tif_name)
        
        
        
        
    else:
        cell_dic = OS_Tools.Load_Variable(cell_method)
        
        
if __name__ == '__main__':
    from My_Wheels.Standard_Parameters.Sub_Graph_Dics import Sub_Dic_Generator
    