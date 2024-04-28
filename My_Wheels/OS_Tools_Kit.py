# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 12:44:56 2019

@author: ZR
This structure is used to do path operations. 


"""
#%%
import os
import pickle
import List_Operation_Kit as List_Tools
import pandas as pd


#%% Function1:Make Dictionary
def mkdir(path,mute = False):
    '''
    
    This function will generate folder at input path. If the folder already exists, then do nothing.\n
    
    Parameters
    ----------
    path : (str)
        Target path you want to generate folder on.
    mute : (bool),optional
        Message will be ignored if mute is True. Default is False
        
    Returns
    -------
    bool
        Whether new folder is generated.

    '''

    isExists=os.path.exists(path)
    if isExists:
        # 如果目录存在则不创建，并提示目录已存在
        if mute == False:
            print('Folder',path,'already exists!')
        return False
    else:
        os.mkdir(path)
        return True
#%% Function2: Get File Name
def Get_File_Name(path,file_type = '.tif',keyword = '',include_sub = False):
    """
    Get all file names of specific type.

    Parameters
    ----------
    path : (str)
        Root path you want to cycle.
    file_type : (str), optional
        File type you want to get. The default is '.tif'.
    keyword : (str), optional
        Key word you need to screen file. Just leave '' if you need all files.
    include_sub : (bool),optional
        If set true, both root folder and subfolder will be cycled.

    Returns
    -------
    Name_Lists : (list)
       Return a list, all file names contained.

    """
    Name_Lists=[]
    for root, dirs, files in os.walk(path):
        for file in files:# walk all files in folder and subfolders.
            if include_sub == False:
                if root == path:# We look only files in root folder, subfolder ignored.
                    if (os.path.splitext(file)[1] == file_type) and (keyword in file):# we need the file have required extend name and keyword contained.
                        Name_Lists.append(os.path.join(root, file))
            else:
                if (os.path.splitext(file)[1] == file_type) and (keyword in file):# we need the file have required extend name and keyword contained.
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
        Loaded file. Same formation as it was saved.If file not exist, return False.

    """
    if file_name == False:
        real_file_path = save_folder
    else:
        real_file_path = save_folder+r'\\'+file_name
    if os.path.exists(real_file_path):
        with open(real_file_path, 'rb') as file:
            loaded_file = pickle.load(file)
        file.close()
    else:
        loaded_file = False
    
    return loaded_file
#%% Function 5: Spike2 Data Reader.
import neo
# updated 230425,MAJOR UPDATE: CHANGE OF NEO API, FOR 0.11.1 version Neo.
# def Spike2_Reader(smr_name,physical_channel = 0):
def Spike2_Reader(smr_name,stream_channel = '0'):
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
    reader = neo.io.Spike2IO(filename=(smr_name),try_signal_grouping=False)
    smr_data = reader.read(lazy=False)[0]
    all_trains = smr_data.segments[0].analogsignals
    
    for i in range(len(all_trains)):
        # current_physics_ch = all_trains[i].annotations['physical_channel_index']
        current_stream_ch = all_trains[i].annotations['stream_id']
        # if physical_channel == current_physics_ch:
        if stream_channel == current_stream_ch:
            fs = all_trains[i].sampling_rate
            channel_data = all_trains[i]
            exported_channel['Capture_Frequent'] = fs
            exported_channel['Channel_Data'] = channel_data
            # exported_channel['Physical_Channel'] = physical_channel
            exported_channel['Stream_Channel'] = stream_channel
            
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
def Get_Subfolders(root_path,keyword = '',method = 'Whole'):
    '''
    Input a path, return sub folders. Absolute path.

    Parameters
    ----------
    root_path : (str)
        The path you want to operate.
    keyword : (str),optional
        If keyword given, only folder have keyword will return.
    method : ('Whole' or 'Relative')
        Determine whether we return only relative file path or the whole path.

    Returns
    -------
    subfolder_paths : (list)
        List of all subfolders. Absolute path provided to simplify usage.

    '''
    all_path = []
    for root, dirs, files in os.walk(root_path):
        if root == root_path:
            for dir_name in dirs:
                if keyword in dir_name:
                    if method == 'Whole':
                        all_path.append(os.path.join(root, dir_name))
                    elif method == 'Relative':
                        all_path.append(dir_name)
    return all_path

#%% Function8, Return upper folder.
def CDdotdot(file_name):
    '''
    Give a file path or folder path, return upper folder, useful to get save folder.

    Parameters
    ----------
    file_name : (str)
        File name or path name.Cannot be ended with '\'

    Returns
    -------
    upper_folder : (str)
        Upper folder or file save folder.
    '''
    file_name = str(file_name)
    upper_folder = '\\'.join(file_name.split('\\')[:-1])
    return upper_folder
#%% Function9, Number_bit_Fill
def Bit_Filler(number,bit_num = 4):
    '''
    This function is used to fill number into specific bits. Fill 0 above.

    Parameters
    ----------
    number : (int or str)
        Number before fill.
    bit_num : (int), optional
        Digital bit you want to fill into. The default is 4.

    Returns
    -------
    filled_num : (str)
        Number filled with 0 above.

    '''
    base_num = pow(10,bit_num)
    added_num = base_num+int(number)
    filled_num = str(added_num)[1:]
    return filled_num


#%% Function10, Memory_Log
def Memory_Logging(save_path,interval = 10):
    import psutil
    import time    
    with open(save_path+r'\\Memory_Log.txt','w') as f:
        while(1):
            c_mem_use = psutil.virtual_memory().used/ 1024 / 1024 / 1024
            print('Current Memory Usage: %.4f GB' % c_mem_use)
            f.write(str(c_mem_use)+' GB Memory Used.\n')
            time.sleep(interval)
#%% Function11, join path in more standart method.
def join(path_A,path_B):
    new_path = os.path.join(path_A,path_B)
    return new_path

#%% Function 12 Another version of load, solving the problem of loading pandas.
def Load_Variable_v2(save_folder,file_name=False):
    if file_name == False:
        real_file_path = save_folder
    else:
        real_file_path = save_folder+r'\\'+file_name
    if os.path.exists(real_file_path):
        pickle_off = open(real_file_path,"rb")
        loaded_file = pd.read_pickle(pickle_off)
        pickle_off.close()
    else:
        loaded_file = False

    
    return loaded_file
    