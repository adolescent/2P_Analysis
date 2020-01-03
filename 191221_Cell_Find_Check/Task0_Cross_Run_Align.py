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
import My_Wheels.Graph_Operation_Kit as Graph_Tools
import numpy as np
import cv2
from My_Wheels.Alignment import Alignment


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
        #Get Run Average First
        self.Before_Align_Dics = {}# Define a dictionary to save before align graphs.
        total_graph_num = 0 # counter of graph numbers
        for i in range(len(self.Before_Align_Tif_Name)):
            run_graph_num = len(self.Before_Align_Tif_Name[i])# How many graphs in this run
            total_graph_num += run_graph_num
            current_run_average = Graph_Tools.Average_From_File(self.Before_Align_Tif_Name[i])
            current_run_average = Graph_Tools.Clip_And_Normalize(current_run_average)
            self.Before_Align_Dics[i] = (current_run_average,run_graph_num) # Write Current dict as a Tuple.
            Graph_Tools.Show_Graph(current_run_average, 'Run_Average',self.all_save_folders[i])# Show and save Run Average.
            
        # Then Use Weighted average method to generate global tif. This methos can be faster than original ones.
        global_average_graph = np.zeros(shape = np.shape(self.Before_Align_Dics[0][0]),dtype = 'f8')# Base on shape of graph
        for i in range(len(self.Before_Align_Tif_Name)):
            global_average_graph += self.Before_Align_Dics[i][0].astype('f8')*self.Before_Align_Dics[i][1]/total_graph_num
        global_average_graph = Graph_Tools.Clip_And_Normalize(global_average_graph)
        
        # At last, save global average graph in every run folders.
        for i in range(len(self.all_save_folders)):
            Graph_Tools.Show_Graph(global_average_graph, 'Global_Average', self.all_save_folders[i],show_time = 0)
        self.Align_Base = global_average_graph # Define Base of Alignment, use this to do the job.
    
    def Align_Cores(self):
        """
        This Function will align every graph and save them in folder 'Aligned Frames'
        
        Returns
        -------
        None.
        """
        for i in range(len(self.Before_Align_Tif_Name)): # Cycle all runs
            for j in range(len(self.Before_Align_Tif_Name[i])): # Cycle current run, between all graph.
                base = self.Align_Base # Use global average as base graph
                current_graph = cv2.imread(self.Before_Align_Tif_Name[i][j],-1)
                _,_,current_aligned_graph = Alignment(base,current_graph) # Calculate aligned graph 
                # Then save graph.
                graph_name = self.Aligned_frame_folders[i]+r'\\'+self.Before_Align_Tif_Name[i][j].split('\\')[-1][:-4]
                Graph_Tools.Show_Graph(current_graph,graph_name,self.Aligned_frame_folders[i],show_time = 0)
                
    def After_Align_Average(self):
        """
        This Functin will generate after align average graph of Run and Global, and then save them.
        
        Returns
        -------
        None.

        """
        self.After_Align_Graphs = {} # Initialize a dictionary, will record all aligned graphs averages and graph nums.
        # Fill After Align Graph Dictionary first
        total_graph_num = 0
        for i in range(len(self.Aligned_frame_folders)):
            current_run_names = OS_Tools.Get_File_Name(self.Aligned_frame_folders)
            temp_average = Graph_Tools.Average_From_File(current_run_names) # This will generate an average graph with 'f8' formation.
            current_graph_aligned = Graph_Tools.Clip_And_Normalize(temp_average)
            Graph_Tools.Show_Graph(current_graph_aligned, 'Run_Average_After_Align', self.all_save_folders[i])
            current_run_Frame_Num = len(current_run_names)
            total_graph_num += current_run_Frame_Num
            self.After_Align_Graphs[i] = (current_graph_aligned,current_run_Frame_Num)
        global_average_after_align = np.zeros(np.shape(current_graph_aligned),dtype = 'f8')
        
        # Then calculate global average in each run.
        for i in range(len(self.all_save_folders)):
            global_average_after_align += self.After_Align_Graphs[i][0]*self.After_Align_Graphs[i][1]/total_graph_num
        global_average_after_align = Graph_Tools.Clip_And_Normalize(global_average_after_align)
        
        # Then save global graph into each folder.
        for i in range(len(self.all_save_folders)):
            if i == 0:
                Graph_Tools.Show_Graph(global_average_after_align, 'Global_Average_After_Align', self.all_save_folders[i])
            else:
                Graph_Tools.Show_Graph(global_average_after_align, 'Global_Average_After_Align', self.all_save_folders[i],show_time = 0)

    
#%% Test Run Here.
        
if __name__ == '__main__':
    
    file_path = [r'E:\ZR\Data_Temp\191215_L77_2P']
    run_name = ['Run01_V4_L11U_D210_GA_RFlocation_shape3_Sti2degStep2deg',
                'Run02_V4_L11U_D210_GA_RFsize',
                'Run03_V4_L11U_D210_GA_RFlocation_shape3_Sti2degStep2deg',
                'Run04_V4_L11U_D210_GA_BACS_ori4_ori8',]
    
    all_folders = List_Op.List_Annex(file_path, run_name)
    CRA = Cross_Run_Align(all_folders)
    CRA.Before_Run_Average()
    CRA.Align_Cores()
    CRA.After_Align_Average()
    
