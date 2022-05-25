# -*- coding: utf-8 -*-
"""
Created on Tue May 24 12:41:57 2022

@author: adolescent

This script will change 220421 file into new caimanned result.
"""


import cv2
from Caiman_API.One_Key_Caiman import One_Key_Caiman
# First, align graphs with Run03.
day_folder = r'D:\Test_Data\2P\220421_L85'
base_graph = cv2.imread(r'D:\Test_Data\2P\220421_L85\_CAIMAN_old\Align_Template.tif',0)
Okc = One_Key_Caiman(day_folder, [1,2,3,7,8,9])
all_runname = Okc.all_stack_names
for i,c_run in enumerate(all_runname):
    Okc.Motion_Corr_Single(c_run,tamplate = base_graph)
    
# Get cell graph from each file.
