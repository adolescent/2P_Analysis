# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 21:12:11 2021

@author: ZR
"""

from My_Wheels.Standard_Aligner import Standard_Aligner

day_folder = r'K:\Test_Data\2P\210202_L76LM_2P'
runlists = list(range(1,12))
SA = Standard_Aligner(day_folder,runlists,final_base = '1-001')
SA.One_Key_Aligner()
