# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 10:23:28 2021

@author: ZR
"""

from Standard_Aligner import Standard_Aligner
day_folder = r'F:\Test_Data\2P\211009_L76_2P'
Sa_1 = Standard_Aligner(day_folder, [1], trans_range=50)
Sa_1.One_Key_Aligner()
Sa_2 = Standard_Aligner(day_folder, [2, 3, 4, 5, 6, 7], trans_range=30, final_base='1-002')
Sa_2.One_Key_Aligner()
