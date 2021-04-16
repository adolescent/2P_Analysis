# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 15:11:48 2021

@author: ZR
"""

from My_Wheels.Standard_Aligner import Standard_Aligner


day_folder = r'K:\Test_Data\2P\210401_L76_2P'
runlists = list(range(1,14))
SA = Standard_Aligner(day_folder,runlists,final_base = '1-011')
SA.One_Key_Aligner()
