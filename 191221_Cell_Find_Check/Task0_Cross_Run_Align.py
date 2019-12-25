# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 16:26:37 2019

@author: ZR
Align All Graphs in different folders, no file/folder reliance.

INPUT : Run Folders
OUTPUT : Aligned Graphs in one standars
         Graph Before&After Align
         Global Average & Run Average

"""


import My_Wheels.OS_Tools_Kit as OTK
import My_Wheels.List_Operation_Kit as LOK

#%%

class Cross_Run_Align(object):
    
    name = r'Align Runs in One Experiments'
    
    def __init__(self,all_folders):
        
        self.all_folders = all_folders
        
        
        
        
        
#%% Test Run Here.
        
if __name__ == '__main__':
    
    file_path = [r'E:\ZR\Data_Temp\191215_L77_2P']
    run_name = ['Run01_V4_L11U_D210_GA_RFlocation_shape3_Sti2degStep2deg',
                'Run02_V4_L11U_D210_GA_RFsize',
                'Run03_V4_L11U_D210_GA_RFlocation_shape3_Sti2degStep2deg',
                'Run04_V4_L11U_D210_GA_BACS_ori4_ori8',]
    
    all_folders = LOK.List_Annex(file_path, run_name)
    
