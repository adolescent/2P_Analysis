# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 14:47:07 2022

@author: ZR
This script is used to find same name as folder A in folder B.
"""

import os


folder_A = r'C:\Users\ZR\Desktop\Test_FolderA'
folder_B = r'C:\Users\ZR\Desktop\Test_FolderB'
splitter = '_' # how to find similar parts.
find_tag = 0



def Get_Sub_Folder(folder_name):
    subfolders = []
    files = os.listdir(folder_name)
    for file in files:
        m = os.path.join(folder_name,file)
        if os.path.isdir(m):
            #subfolders.append(m)
            subfolders.append(file)
    return subfolders

#%% Get Subfolder Names
A_folders = Get_Sub_Folder(folder_A)
B_folders = Get_Sub_Folder(folder_B)

duplicated_tag_name = []
for c_path in A_folders:
    c_tager = c_path.split(splitter)[find_tag]
    for c_B_path in B_folders:
        if c_tager in c_B_path:
            print(c_path+' same as '+c_B_path)
            duplicated_tag_name.append(c_tager)
