# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 20:03:50 2020

@author: zhang
"""

target_root_path = r'C:\Users\ZR\Desktop\可移动磁盘'
mode = 'left'
ignore_level = 1 # which level of path will be ignored.


import os
import pandas as pd
from tqdm import tqdm

root_iter = len(target_root_path.split('\\'))
max_iter = 0
all_file_names = []
for root,dirs,files in os.walk(target_root_path):
    for file in files:
        # get path name.
        #print(root)
        #print(os.path.join(root,file))
        c_path = os.path.join(root,file)
        c_iter = len(c_path.split('\\'))
        max_iter = max(max_iter,c_iter)
        all_file_names.append(c_path)    
# write into pandas frame.
path_frame = pd.DataFrame('',columns = range(max_iter-root_iter),index = range(len(all_file_names)))
for i,cp in tqdm(enumerate(all_file_names)):
    seperated_path = cp.split('\\')[root_iter:]
    if mode == 'right':
        for j in range(-1,-1-len(seperated_path),-1):# iter from different sequence.
            path_frame.iloc[i,j] = seperated_path[j]
    elif mode == 'left':
        for j in range(len(seperated_path)):# iter from different sequence.
            path_frame.iloc[i,j] = seperated_path[j]
# cut path if it's not needed.
if ignore_level != 0:
    path_frame = path_frame.iloc[:,:-ignore_level]
    path_frame = path_frame.drop_duplicates()

path_frame.to_csv(index = False,header = False,path_or_buf='All_File_Name.csv')




# csv test
# =============================================================================
# df = pd.DataFrame({'name': ['Raphael', 'Donatello',''],
#                    'mask': ['red', 'purple','Tesr'],
#                    'weapon': ['sai', 'bo staff','']})
# df.to_csv(index = False,path_or_buf='test.csv')
# =============================================================================




