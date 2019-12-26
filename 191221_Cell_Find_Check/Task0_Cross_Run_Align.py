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


import My_Wheels.OS_Tools_Kit as OS_Tools
import My_Wheels.List_Operation_Kit as List_Op



class Cross_Run_Align(object):
    '''Cross All Runs in one day, with the same base.
    Actually we use this part as a function, package this into a class will be useful in function extention.
    '''
    
    name = r'Align Runs in One Experiments'
    
    def __init__(self,all_folders):
        
        self.all_folders = all_folders
        self.all_save_folders = List_Op.List_Annex(self.all_folders,['Results'])
        self.Aligned_frame_folders = List_Op.List_Annex(self.all_save_folders,['Aligned_Frames'])
        for i in range(len(self.all_save_folders)):
            OS_Tools.mkdir(self.all_save_folders[i])
            OS_Tools.mkdir(self.Aligned_frame_folders[i])
        self.Before_Align_Tif_Name = []
        for i in range(len(self.all_folders)):
            current_run_tif = OS_Tools.Get_File_Name(self.all_folders[i])
            self.Before_Align_Tif_Name.append(current_run_tif)
        
    def Before_Run_Average(self):
        """
        Generate global and per run average graph.
        This part is automatic,output averaged graph in folder, return nothing.
        Returns
        -------
        None.
        """
        all_tif_name_before = []
        for i in range(len(self.Before_Align_Tif_Name)):
            all_tif_name_before.extend(self.Before_Align_Tif_Name[i])
            
        
        
        
#%% Test Run Here.
        
if __name__ == '__main__':
    
    file_path = [r'E:\ZR\Data_Temp\191215_L77_2P']
    run_name = ['Run01_V4_L11U_D210_GA_RFlocation_shape3_Sti2degStep2deg',
                'Run02_V4_L11U_D210_GA_RFsize',
                'Run03_V4_L11U_D210_GA_RFlocation_shape3_Sti2degStep2deg',
                'Run04_V4_L11U_D210_GA_BACS_ori4_ori8',]
    
    all_folders = List_Op.List_Annex(file_path, run_name)
    CRA = Cross_Run_Align(all_folders)
