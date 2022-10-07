# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 14:29:27 2022

@author: ZR

This will duplicate folder(and files included) with specific tag into new folder.

Maintain subfolder name unchanged.
"""

import os
import shutil
from tqdm import tqdm

base_folder = r'C:\Users\ZR\Desktop\Test_FolderB'
save_folder = r'C:\Users\ZR\Desktop\新建文件夹'
tag = 'afg'

def Get_Sub_Folder(folder_name):
    subfolders = []
    files = os.listdir(folder_name)
    for file in files:
        m = os.path.join(folder_name,file)
        if os.path.isdir(m):
            #subfolders.append(m)
            subfolders.append(file)
    return subfolders

# 1.Get for-copy subfolder. 
all_subfolder = Get_Sub_Folder(base_folder)
usable_subfolder = []
for c_subfolder in all_subfolder:
    if tag in c_subfolder:
        usable_subfolder.append(c_subfolder)
# 2.copy subfolders into destination path, remain name unchanged.
for c_subfolder in tqdm(usable_subfolder):
    whole_path_tar = os.path.join(base_folder,c_subfolder)
    whole_path_dest = os.path.join(save_folder,c_subfolder)
    shutil.copytree(whole_path_tar, whole_path_dest)