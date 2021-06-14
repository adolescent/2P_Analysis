# -*- coding: utf-8 -*-
"""
Created on Fri May 21 14:08:26 2021

@author: ZR
"""
from Spontaneous_Processor import Spontaneous_Processor


day_folder = r'K:\Test_Data\2P\210423_L76_2P'
SP_After = Spontaneous_Processor(day_folder,spon_run = r'Run015')
SP_After.Do_PCA(0,9999)
SP_Before = Spontaneous_Processor(day_folder,spon_run = r'Run001')
SP_Before.Do_PCA(0,2600)
