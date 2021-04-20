# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 10:24:11 2021

@author: ZR
"""

from My_Wheels.Standard_Aligner import Standard_Aligner

day_folder = r'K:\Test_Data\2P\210320_L76_2P'
runlists = list(range(1,16))
SA = Standard_Aligner(day_folder,runlists,final_base = '1-002')
SA.One_Key_Aligner()
