# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 12:44:56 2019

@author: ZR
This structure is used to do path operations. 


"""

import os
import pickle
import List_Operation_Kit as List_Tools

#%% Function1:Make Dictionary
def mkdir(path):
    '''
    
    This function will generate folder at input path. If the folder already exists, then do nothing.\n
    
    Parameters
    ----------
    path : (str)
        Target path you want to generate folder on.
        
    Returns
    -------
    bool
        Whether new folder is generated.

    '''

    isExists=os.path.exists(path)
    if isExists:
        # 如果目录存在则不创建，并提示目录已存在
        print('Folder',path,'already exists!')
        return False
    else:
        os.mkdir(path)
        return True
#%% Function2: Get File Name
def Get_File_Name(path,file_type = '.tif'):
    """
    Get all file names of specific type.

    Parameters
    ----------
    path : (str)
        Root path you want to cycle.
    file_type : (str), optional
        File type you want to get. The default is '.tif'.

    Returns
    -------
    Name_Lists : (list)
       Return a list, all file names contained.

    """
    Name_Lists=[]
    for root, dirs, files in os.walk(path):
        for file in files:
            if root == path:#只遍历根目录，不操作子目录的文件
                if os.path.splitext(file)[1] == file_type:
                    Name_Lists.append(os.path.join(root, file))
                    
    return Name_Lists
#%% Function3: Save a Variable to binary data type.
def Save_Variable(save_folder,name,variable,extend_name = '.pkl'):
    """
    Save a variable as binary data.

    Parameters
    ----------
    save_folder : (str)
        Save Path. Only save folder.
    name : (str)
        File name.
    variable : (Any Type)
        Data you want to save.
    extend_name : (str), optional
        Extend name of saved file. The default is '.pkl'.

    Returns
    -------
    bool
        Nothing.

    """
    if os.path.exists(save_folder):
        pass 
    else:
        os.mkdir(save_folder)
    
    real_save_path = save_folder+r'\\'+name+extend_name
    fw = open(real_save_path,'wb')
    pickle.dump(variable,fw)
    fw.close()
    
    return True
#%% Function 4: Load saved binary file.
def Load_Variable(save_folder,file_name=False):
    """
    Load variable from file. Single value input supported.

    Parameters
    ----------
    save_folder : (str)
        Folder of files.
    file_name : (str)
        Name of file. Extend name shall be contained! If you want to input single value path, just ignore this part.

    Returns
    -------
    loaded_file : (Any type)
        Loaded file. Same formation as it was saved.

    """
    if file_name == False:
        real_file_path = save_folder
    else:
        real_file_path = save_folder+r'\\'+file_name
    with open(real_file_path, 'rb') as file:
        loaded_file = pickle.load(file)
    file.close()
    
    return loaded_file
#%% Function 5: Spike2 Data Reader.
import neo
def Spike2_Reader(smr_name,physical_channel = 0):
    """
    
    Export a single channel from .smr data

    Parameters
    ----------
    smr_name : (str)
        Smr file name. extend name shall be contained
    physical_channel : (int), optional
        COM Port of 1401. The default is 0.
        Traditionally, port 0 is ViSaGe input, port3 is 2P Scanning Wave.
    Returns
    -------
    exported_channel : (Dic)
        Exported channel data.
        ['Capture_Frequent'] = Capture frequency, Hz
        ['Channel_Data'] = time_series
        ['Physical_Channel'] = Physical Channel of 1401.

    """
    
    exported_channel = {}
    reader = neo.io.Spike2IO(filename=(smr_name))
    smr_data = reader.read(lazy=False)[0]
    all_trains = smr_data.segments[0].analogsignals
    
    for i in range(len(all_trains)):
        current_physics_ch = all_trains[i].annotations['physical_channel_index']
        if physical_channel == current_physics_ch:
            fs = all_trains[i].sampling_rate
            channel_data = all_trains[i]
            exported_channel['Capture_Frequent'] = fs
            exported_channel['Channel_Data'] = channel_data
            exported_channel['Physical_Channel'] = physical_channel
            
    return exported_channel
#%% Function 6: Get Last Saved file name.
import time
import numpy as np
def Last_Saved_name(path,file_type = '.txt'):
    """
    Return last saved file name. Only usable at the same day!

    Parameters
    ----------
    path : (str)
        Target data folder.
    file_type : (str)
        Which type of data you want to find.

    Returns
    -------
    name : (str)
        Last saved data.

    """
    all_file_name = Get_File_Name(path,file_type)
    all_file_time = np.array([])
    for i in range(len(all_file_name)):
        current_file_time = int(time.strftime("%H%M%S",time.localtime(os.stat(all_file_name[i]).st_mtime)))
        all_file_time = np.append(all_file_time,current_file_time)
    last_file_name = all_file_name[np.where(all_file_time == np.max(all_file_time))[0][0]]
    return last_file_name

#%% Function7 Get sub folders name
def Get_Sub_Folders(current_folder):
    '''
    Input a path, return sub folders. Absolute path.

    Parameters
    ----------
    current_folder : (str)
        The path you want to operate.

    Returns
    -------
    sub_folders : (list)
        List of all subfolders. Absolute path provided to simplify usage.

    '''
    all_subfolders = os.listdir(current_folder)
    for i in range(len(all_subfolders)-1,-1,-1):
        current_sf = all_subfolders[i]
        # Then remove file names
        if len(current_sf.split('.')) == 2:# meaning this is a file, have extend name.
            all_subfolders.pop(i)
    sub_folders = List_Tools.List_Annex([current_folder], all_subfolders)
    return sub_folders