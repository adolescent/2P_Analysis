# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 10:07:29 2021

@author: ZR

Based on Final Averaged graphs and mannual cells(not necessary).
"""

import OS_Tools_Kit as ot
import Graph_Operation_Kit as gt
import List_Operation_Kit as lt
import cv2
from Spike_Train_Generator import Spike_Train_Generator
from Spike_Train_Generator import Single_Condition_Train_Generator


class Standard_Cell_Generator(object):
    name = 'Standard Cell Data Generator'
    def __init__(self,animal_name,
                 date,
                 day_folder,
                 runid_lists,
                 location = 'A',
                 Stim_Frame_Align_name = '_All_Stim_Frame_Infos.sfa',
                 cell_subfolder = r'\_Manual_Cell'                 
                 ):
        print('Align,Cell Find,Stim Frame Align shall be done before.')
        self.save_folder = day_folder
        self.cell_prefix = animal_name+'_'+date+location+'_'
        self.all_SFA_dic = ot.Load_Variable(day_folder+'\\'+Stim_Frame_Align_name)
        cell_path = ot.Get_File_Name(day_folder+cell_subfolder,'.cell')[0]
        self.cell_infos = ot.Load_Variable(cell_path)['All_Cell_Information']
        self.cell_num = len(self.cell_infos)
        self.all_runnames = []
        for i in range(len(runid_lists)):
            c_runid = runid_lists[i]
            self.all_runnames.append('Run'+str(ot.Bit_Filler(c_runid,3)))
        self.all_runsubfolders = lt.List_Annex([day_folder], lt.Run_Name_Producer_2P(runid_lists))
        
    
    def Cell_Struct_Generator(self,mask = r'\Location_Mask.tif'):
        self.All_Cells = {} # output variable,including all cell informations.
        for i in range(self.cell_num):
            c_cell_name = self.cell_prefix+ot.Bit_Filler(i,4)
            self.All_Cells[c_cell_name] = {}# single cell working space.
            self.All_Cells[c_cell_name]['Name'] = c_cell_name
            self.All_Cells[c_cell_name]['Cell_Info'] = self.cell_infos[i]
            self.All_Cells[c_cell_name]['Cell_Area'] = self.cell_infos[i].convex_area
            # Then we determine whether this cell in each run.
            self.All_Cells[c_cell_name]['In_Run'] = {}
            for j in range(len(self.all_runsubfolders)):
                c_sp = self.all_runsubfolders[j]+'\Results'
                c_mask = cv2.imread(c_sp+mask,-1)
                c_mask = c_mask>(c_mask/2)
                
                cell_location = self.cell_infos[i].coords
                inmask_area = c_mask[cell_location[:,0],cell_location[:,1]].sum()
                if inmask_area == self.cell_infos[i].area:
                    self.All_Cells[c_cell_name]['In_Run'][self.all_runnames[j]] = True
                    self.All_Cells[c_cell_name][self.all_runnames[j]] = {}
                else:
                    self.All_Cells[c_cell_name]['In_Run'][self.all_runnames[j]] = False
            self.all_cellnames = list(self.All_Cells.keys())
    
    
    def Firing_Trains(self,align_subfolder = r'\Results\Final_Aligned_Frames'):
        # cycle all runs
        for i in range(len(self.all_runsubfolders)):
            all_tif_name = ot.Get_File_Name(self.all_runsubfolders[i]+align_subfolder)
            c_F_train,c_dF_F_train = Spike_Train_Generator(all_tif_name, self.cell_infos)
            # seperate trains into cell if in run.
            c_runname = self.all_runnames[i]
            for j in range(len(self.all_cellnames)):
                c_cellname = self.all_cellnames[j]
                if c_runname in self.All_Cells[c_cellname]:# meaning we have this run
                    self.All_Cells[c_cellname][c_runname]['F_train'] = c_F_train[j]
                    self.All_Cells[c_cellname][c_runname]['dF_F_train'] = c_dF_F_train[j]
    
    def Condition_Data(self,response_extend = (3,3),
                       base_frame = [0,1,2],
                       filter_para = (0.02,False),
                       ROI_extend = (7,7),
                       ROI_base_frame = [0,1,2,3,4],
                       ROI_filter_para = (0.01,False),
                       full_size = (512,512)):
        
        # cycle all runs
        for i in range(len(self.all_runnames)):
            c_runname = self.all_runnames[i]
            examp_graph = cv2.imread(ot.Get_File_Name(self.all_runsubfolders[i])[0],-1)
            if examp_graph.shape == full_size:
                is_ROI = False
            else:
                is_ROI = True
            # Cycle all cells
            for j in range(len(self.all_cellnames)):
                c_cell_dic = self.All_Cells[self.all_cellnames[j]]
                if (c_runname in c_cell_dic) and (self.all_SFA_dic[c_runname] != None):# This cell in in this run and not spon.
                    t_F_train = c_cell_dic[c_runname]['F_train']
                    if is_ROI :
                        t_CR_Train,t_Raw_CR_Train = Single_Condition_Train_Generator(t_F_train, 
                                                                                     self.all_SFA_dic[c_runname],
                                                                                     ROI_extend[0],ROI_extend[1],
                                                                                     ROI_base_frame,ROI_filter_para)
                    else:
                        t_CR_Train,t_Raw_CR_Train = Single_Condition_Train_Generator(t_F_train, 
                                                                                     self.all_SFA_dic[c_runname],
                                                                                     response_extend[0],response_extend[1],
                                                                                     base_frame,filter_para)
                    self.All_Cells[self.all_cellnames[j]][c_runname]['CR_Train'] = t_CR_Train
                    self.All_Cells[self.all_cellnames[j]][c_runname]['Raw_CR_Train'] = t_Raw_CR_Train
                    

    def Generate_Cells(self):
        self.Cell_Struct_Generator()
        self.Firing_Trains()
        self.Condition_Data()
        ot.Save_Variable(self.save_folder, self.cell_prefix+'_All_Cells',self.All_Cells,'.ac')