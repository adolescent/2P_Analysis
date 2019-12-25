# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 12:44:56 2019

@author: ZR
This structure is used to do path operations. 


"""

import os


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
    Get all file names of specific file type, 

    Parameters
    ----------
    path : (str)
        Root path you want to path.
    file_type : TYPE, optional
        DESCRIPTION. The default is '.tif'.

    Returns
    -------
    Name_Lists : TYPE
        DESCRIPTION.

    """
    
    return Name_Lists
    
