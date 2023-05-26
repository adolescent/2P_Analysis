# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 12:03:23 2022

@author: ZR

Rename folder with given name. 

"""

import os
import shutil

root_folder = r'C:\Users\ZR\Desktop\test_folder'
tag_anno = '[asdf]AA0'
change_into = 'Try_This_One[aa]_'

# get all subfolder, only 1 level.
subfolders = []
files = os.listdir(root_folder)
for file in files:
    m = os.path.join(root_folder,file)
    if os.path.isdir(m):
        #subfolders.append(m)
        subfolders.append(file)
        
# Then change subfoler have specific tag into other mode.
changed_subfolders = []
for c_path in subfolders:
    if tag_anno in c_path:
        print(c_path)
        new_path = c_path.replace(tag_anno,change_into)
        changed_subfolders.append(new_path)
        origin_folder = os.path.join(root_folder, c_path)
        tar_folder = os.path.join(root_folder, new_path)
        shutil.move(origin_folder,tar_folder)





