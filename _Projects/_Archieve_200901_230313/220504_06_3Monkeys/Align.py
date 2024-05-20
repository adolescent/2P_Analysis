# -*- coding: utf-8 -*-
"""
Created on Mon May  9 10:17:14 2022

@author: ZR
"""

from Standard_Aligner import Standard_Aligner

day_folder = r'D:\ZR\_Temp_Data\220506_L76_2P'
Sa = Standard_Aligner(day_folder, [1,2,3,4,5,6,7],final_base = '1-003')
Sa.One_Key_Aligner_No_Affine()

