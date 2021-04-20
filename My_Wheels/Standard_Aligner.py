# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 14:51:38 2021

@author: ZR
Align each run first, then match them use affine method.
ROI can be operated in this too. Boulder will fill in blank.

"""

import List_Operation_Kit as lt
import OS_Tools_Kit as ot
import Graph_Operation_Kit as gt
from Translation_Align_Function import Translation_Alignment
import cv2
from Tremble_Evaluator import Least_Tremble_Average_Graph
from Affine_Alignment import Affine_Aligner_Gaussian
from Affine_Alignment import Affine_Core_Point_Equal
import os
import numpy as np
import shutil

class Standard_Aligner(object):
    
    name = 'A Complex combined Aligner.'
    
    def __init__(self,day_folder,runids,final_base = '1-002',
                 trans_range = 35,full_size = (512,512)):
        self.full_size = full_size
        self.run_subfolder = lt.Run_Name_Producer_2P(runids)
        self.all_runfolders = lt.List_Annex([day_folder], self.run_subfolder)
        self.runnum = len(self.all_runfolders)
        self.trans_range = trans_range
        targ_ind = self.run_subfolder.index(final_base)
        self.target_runfolder = self.all_runfolders[targ_ind]
        self.all_resultfolder = lt.List_Annex(self.all_runfolders, ['Results'])
        self.is_ROI = {}
        self.graph_size = {}
        for i in range(self.runnum):
            example = cv2.imread(ot.Get_File_Name(self.all_runfolders[i])[0],-1)
            self.graph_size[self.run_subfolder[i]] = example.shape
            if example.shape != full_size:
                self.is_ROI[self.run_subfolder[i]] = True
            else:
                self.is_ROI[self.run_subfolder[i]] = False
        # check base is full frame.
        if self.graph_size[final_base] != full_size:
            raise IOError('Base must be full frame.')

    def Seperate_Translation_Align(self):
        for i in range(self.runnum):
            Translation_Alignment([self.all_runfolders[i]],
                                  align_range = self.trans_range,
                                  align_boulder= self.trans_range,
                                  graph_shape = self.graph_size[self.run_subfolder[i]],timer = False
                                  )
        self.all_translation_folder = lt.List_Annex(self.all_resultfolder, ['Aligned_Frames'])

    def Affine_Realign(self):
        # Only do this in full frame, ROI will skip.
        for i in range(self.runnum):
            c_run = self.run_subfolder[i]
            if self.is_ROI[c_run] == False:
                c_average_graph,_ = Least_Tremble_Average_Graph(self.all_translation_folder[i])
                c_average_graph = gt.Clip_And_Normalize(c_average_graph,clip_std = 5)
                gt.Show_Graph(c_average_graph, 'Affine_Base',self.all_resultfolder[i])
                Affine_Aligner_Gaussian(self.all_translation_folder[i],
                                        c_average_graph,
                                        save_folder= self.all_resultfolder[i])
    def Cross_Run_Align(self):
        # Align each average graph, and use translation matrix to all frames.
        # All frames aligned to final base
        final_base = cv2.imread(self.target_runfolder+r'\Results\Graph_After_Affine.tif',-1)
        # Target run just change foldername.
        os.rename(self.target_runfolder+r'\Results\Affined_Frames',self.target_runfolder+r'\Results\Final_Aligned_Frames')
        targ_mask = np.ones(shape = self.full_size)
        gt.Show_Graph((targ_mask*65535).astype('u2'), 'Location_Mask', self.target_runfolder+r'\Results')
        # And generate full frame mask.
        # other run will use average match to realign.
        other_runs = list(self.all_runfolders)
        other_runs.remove(self.target_runfolder)
        for i in range(len(other_runs)):
            # Treat ROI and non ROI seperately.
            if self.is_ROI[other_runs[i].split('\\')[-1]]:# if ROI, use aligned frame as base.
                c_average_graph = cv2.imread(other_runs[i]+r'\Results\Run_Average_After_Align.tif',-1)
                c_atn = ot.Get_File_Name(other_runs[i]+r'\Results\Aligned_Frames')    
            else:
                c_average_graph = cv2.imread(other_runs[i]+r'\Results\Graph_After_Affine.tif',-1)
                c_atn = ot.Get_File_Name(other_runs[i]+r'\Results\Affined_Frames')    
            c_matched_graph,c_h =  Affine_Core_Point_Equal(c_average_graph, final_base,targ_gain = 1)
            # Location mask, maybe used later
            mask = np.ones(shape = c_average_graph.shape)
            resized_mask = cv2.warpPerspective(mask, c_h, (self.full_size[1],self.full_size[0]))
            gt.Show_Graph((resized_mask*65535).astype('u2'), 'Location_Mask', other_runs[i]+r'\Results')
            # Combined graph, for record
            resized_average = cv2.warpPerspective(c_average_graph, c_h,(self.full_size[1],self.full_size[0]))
            combined_graph = cv2.cvtColor(final_base,cv2.COLOR_GRAY2RGB).astype('f8')
            combined_graph[:,:,1] += resized_average
            combined_graph = np.clip(combined_graph,0,65535).astype('u2')
            gt.Show_Graph(combined_graph, 'Combined_Location_Graph', other_runs[i]+r'\Results')
            # Then get final affined graph, based on former 
            final_align_folder = other_runs[i]+r'\Results\Final_Aligned_Frames'
            ot.mkdir(final_align_folder)
            for j in range(len(c_atn)):
                c_graph_name = c_atn[j].split('\\')[-1][:-4]
                c_graph = cv2.imread(c_atn[j],-1)
                aligned_c_graph = cv2.warpPerspective(c_graph, c_h, (self.full_size[1],self.full_size[0]))
                gt.Show_Graph(aligned_c_graph, c_graph_name, final_align_folder,show_time=0)
    
        #Last generate after average graph.
        for i in range(len(self.all_resultfolder)):
            after_atn = ot.Get_File_Name(self.all_resultfolder[i]+r'\Final_Aligned_Frames')
            after_average = gt.Clip_And_Normalize(gt.Average_From_File(after_atn),clip_std=5)
            gt.Show_Graph(after_average, 'Final_Averaged_Graph', self.all_resultfolder[i])
    
    def Delete_Middle_Folders(self):
        print('All middle averag will be deleted. Be cautious!')
        for i in range(len(self.all_resultfolder)):
            shutil.rmtree(self.all_resultfolder[i]+r'\Aligned_Frames')
            have_affine = os.path.exists(self.all_resultfolder[i]+r'\Affined_Frames')
            if have_affine:
                shutil.rmtree(self.all_resultfolder[i]+r'\Affined_Frames')
                
    def Get_Final_Average(self):
        all_averaged_tif_name = []
        for i in range(len(self.run_subfolder)):
            if self.is_ROI[self.run_subfolder[i]] == False: # only average not ROI frames.
                all_averaged_tif_name.extend(ot.Get_File_Name(self.all_resultfolder[i]+r'\Final_Aligned_Frames'))
        total_averaged_graph = gt.Clip_And_Normalize(gt.Average_From_File(all_averaged_tif_name),clip_std=5)
        for i in range(len(self.all_resultfolder)):
            if i ==0:
                gt.Show_Graph(total_averaged_graph, 'Global_Average', self.all_resultfolder[i])
            else:
                gt.Show_Graph(total_averaged_graph, 'Global_Average', self.all_resultfolder[i],show_time = 0)
            
    
    def One_Key_Aligner(self):
        '''
        One key align function, all work will be done automatically.
        CAUTION:If Tremble is too big, this might cause problem.
        '''
        print('Start Aligning to each run..')
        self.Seperate_Translation_Align()
        print('Affine based on each average graphs')
        self.Affine_Realign()
        print('Cross run aligning')
        self.Cross_Run_Align()
        print('Jobs done, generating averag graph.')
        self.Get_Final_Average()
        self.Delete_Middle_Folders()





    
    
    