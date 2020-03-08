# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 12:44:56 2019

@author: ZR
This structure is used to do path operations. 


"""

import os
import pickle

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
    real_save_path = save_folder+r'\\'+name+extend_name
    fw = open(real_save_path,'wb')
    pickle.dump(variable,fw)
    fw.close()
    
    return True
#%% Function 4: Load saved binary file.
def Load_Variable(save_folder,file_name):
    """
    Load variable from file.

    Parameters
    ----------
    save_folder : (str)
        Folder of files.
    file_name : (str)
        Name of file. Extend name shall be contained!

    Returns
    -------
    loaded_file : (Any type)
        Loaded file. Same formation as it was saved.

    """
    	
    real_file_path = save_folder+r'\\'+file_name
    with open(real_file_path, 'rb') as file:
        loaded_file = pickle.load(file)
    file.close()
    
    return loaded_file
#%% Function 5: Spike2 Data Reader.
import neo
def Spike2_Reader(smr_name,smr_path,physical_channel = 0):
    """
    
    Export a single channel from .smr data

    Parameters
    ----------
    smr_name : (str)
        Smr file name. extend name shall be contained
    smr_path : (str)
        Smr File Path. Path only.
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
    reader = neo.io.Spike2IO(filename=(smr_path+r'\\'+smr_name))
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