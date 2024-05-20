# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 19:06:59 2022

@author: ZR
"""

from Caiman_API.Precess_Pipeline import Preprocess_Pipeline

day_folder = r'D:\ZR\_Temp_Data\220825_L85_2P'

pp = Preprocess_Pipeline(day_folder, [1,2,3,6,7],boulder = (20,30,20,40))
pp.Do_Preprocess()
