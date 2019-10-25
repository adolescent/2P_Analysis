# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 13:38:11 2019

@author: ZR
"""

import os

class Path_Control(object):
    
    name = r'输入和输出操作，包括文件夹操作在内'
    
    def mkdir(path):
        isExists=os.path.exists(path)
        if isExists:
            # 如果目录存在则不创建，并提示目录已存在
            print('Folder',path,'already exists!')
            return False
        else:
            os.mkdir(path)
            return True
        
    def file_name(file_dir,file_type = '.tif'):#读取当前目录下的同拓展名的所有文件名，并返回为一个列表L。 
        L=[] 
        for root, dirs, files in os.walk(file_dir):
            for file in files:
                if root == file_dir:#只遍历根目录，不操作子目录的文件
                    if os.path.splitext(file)[1] == file_type:
                        L.append(os.path.join(root, file))
        return L
    
    
    
    