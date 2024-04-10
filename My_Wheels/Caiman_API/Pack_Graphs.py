# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 14:01:49 2022

@author: ZR
"""
from Decorators import Timer
import OS_Tools_Kit as ot
import tifffile as tif
import numpy as np
import List_Operation_Kit as lt
import cv2

@Timer
def Graph_Packer(data_folder_lists,save_folder,graph_shape = (512,512)):
    '''
    Pack all tif files in data folder into a tif stack.

    Parameters
    ----------
    data_folder : (list)
        List of folder of original data.
    save_folder : (str)
        Folder to save graphs into.
        
    Return
    ----------
    frame_num_lists : (list)
        List of frame numbers given. This is used to cut series back.
    
    '''
    
    frame_num_lists = []
    for i,c_folder in enumerate(data_folder_lists):
        c_tif_name = ot.Get_File_Name(c_folder)
        frame_num_lists.append(len(c_tif_name))
        c_tif_struct = np.zeros(shape = (len(c_tif_name),graph_shape[0],graph_shape[1]),dtype='u2')
        c_folder_name = c_folder.split('\\')[-1]
        for i in range(len(c_tif_name)):
            c_graph = cv2.imread(c_tif_name[i],-1)
            c_tif_struct[i,:,:] = c_graph
        tif.imwrite(save_folder+r'\\'+c_folder_name+'.tif',c_tif_struct)

    return frame_num_lists

@Timer
def Graph_Packer_Cut(data_folder_lists,save_folder,graph_shape = (512,512),cutsize = 1500,min_frame_num = 500):
    '''
    
    Pack all tif frames in data folder,and cut them into small packs. This is used to avoid 4GB problem.
    
    Parameters
    ----------
    data_folder : (list)
        List of folder of original data.
    save_folder : (str)
        Folder to save graphs into.
    graph_shape : (turple)
        Shape of graphs.
    cut_size : (int,optional)
        Frame number of each tif stack. Default is 1500.
    min_frame_num : (int,optional)
        Smallest number of last frame. Default is 500.
        
    Return
    ----------
    frame_num_lists : (list)
        List of frame numbers given. This is used to cut series back.
    run_name_dic : (dic)
        Dictionary of run-compared tif frames. Some run may be cutted into multiple tif file.
        

    '''
    
    frame_num_lists = []
    run_name_dic = {}
    for i,c_folder in enumerate(data_folder_lists):
        c_tif_name = ot.Get_File_Name(c_folder)
        frame_num_lists.append(len(c_tif_name))
        # cut current run into several subfiles.
        last_frame_num = len(c_tif_name)%cutsize
        if last_frame_num>min_frame_num:# if last frame number <100, concat them into file before.
            subfile_num = np.ceil(len(c_tif_name)/cutsize).astype('int')
        else:
            subfile_num = np.round(len(c_tif_name)/cutsize).astype('int')
        c_folder_name = c_folder.split('\\')[-1]
        run_name_dic[c_folder_name] = []
        # except last one
        for j in range(subfile_num-1):
            if j<9:
                c_filename = c_folder_name+'_0'+str(j+1)+'.tif'
            else:
                c_filename = c_folder_name+'_'+str(j+1)+'.tif'
            whole_c_filename = ot.join(save_folder,c_filename)
            run_name_dic[c_folder_name].append(whole_c_filename)
            c_tif_struct = np.zeros(shape = (cutsize,graph_shape[0],graph_shape[1]),dtype='u2')
            for k in range(j*cutsize,(j+1)*cutsize):
                c_graph = cv2.imread(c_tif_name[k],-1)
                c_tif_struct[k%cutsize,:,:] = c_graph
            tif.imwrite(whole_c_filename,c_tif_struct)
        # add last one.
        if subfile_num<10:
            last_filename = c_folder_name+'_0'+str(subfile_num)+'.tif'
        else:
            last_filename = c_folder_name+'_'+str(subfile_num)+'.tif'
        whole_last_filename = ot.join(save_folder,last_filename) 
        run_name_dic[c_folder_name].append(whole_last_filename)
        c_tif_struct = np.zeros(shape = (len(c_tif_name)-cutsize*(subfile_num-1),graph_shape[0],graph_shape[1]),dtype='u2')
        for j in range(cutsize*(subfile_num-1),len(c_tif_name)):
            c_graph = cv2.imread(c_tif_name[j],-1)
            c_tif_struct[j%cutsize,:,:] = c_graph
        tif.imwrite(whole_last_filename,c_tif_struct)

    return frame_num_lists,run_name_dic

def Count_Frame_Num(data_folder_lists):
    '''
    Count frame number of given lists.

    Parameters
    ----------
    data_folder_lists : (list)
        List of data folder.

    Returns
    -------
    frame_num_lists : (list)
        List of frame number in each run.

    '''
    
    frame_num_lists = []
    run_subnames = []
    for i,c_folder in enumerate(data_folder_lists):
        c_tif_name = ot.Get_File_Name(c_folder)
        frame_num_lists.append(len(c_tif_name))
        run_subnames.append(c_folder.split('\\')[-1])

    return frame_num_lists


def Get_Runname_Dic(run_lists,save_folder):
    
    run_subfolders = lt.Run_Name_Producer_2P(run_lists)
    run_name_dic = {}
    all_tif_name = ot.Get_File_Name(save_folder)
    for i,c_run in enumerate(run_subfolders):
        run_name_dic[c_run] = []
        for j,c_tifname in enumerate(all_tif_name):
            if c_run in c_tifname:
                run_name_dic[c_run].append(c_tifname)
                
    return run_name_dic
    