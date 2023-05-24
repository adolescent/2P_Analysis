# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 13:51:07 2022

@author: adolescent

Generate all pre process in one key.

"""
#%%
from Caiman_API.One_Key_Caiman import One_Key_Caiman
import OS_Tools_Kit as ot
from Stim_Frame_Align import One_Key_Stim_Align
from Caiman_API.Condition_Response_Generator import All_Cell_Condition_Generator
from Caiman_API.Map_Generators_CAI import One_Key_T_Map
from Stimulus_Cell_Processor.Get_Cell_Tuning_Cai import Tuning_Calculator
import warnings
from Decorators import Timer

class Preprocess_Pipeline(object):
    
    name = r'One key preprocess'
    
    def __init__(self,day_folder,runlist,max_shift = (75,75),
                 boulder = (20,20,20,20),in_server = True,align_base = '1-003',
                 od_run = 'Run006',od_type = 'OD_2P',
                 orien_run = 'Run002',orien_type = 'G16_2P',
                 color_run = 'Run007',color_type = 'HueNOrien4',
                 ):
        warnings.filterwarnings("ignore")
        self.day_folder = day_folder
        self.boulder = boulder
        self.runlist = runlist
        self.in_server = in_server
        self.align_base = align_base
        # get stimuli folder.
        all_runfolders = ot.Get_Sub_Folders(self.day_folder)
        for i,c_folder in enumerate(all_runfolders):
            if 'stimuli' in c_folder:
                self.stim_folder = c_folder
        # make sub paras standarized.
        self.od_run = od_run
        self.od_type = od_type
        self.orien_run = orien_run
        self.orien_type = orien_type
        self.color_run = color_run
        self.color_type = color_type
        self.max_shift = max_shift
        
    @Timer
    def Do_Preprocess(self):
        # do stim frame align
        if ot.Get_File_Name(self.day_folder,'.sfa') != []:
            print('Stim Frame Align Already Done.')
        else:
            print('Stim Frame Align First...\n')
            One_Key_Stim_Align(self.stim_folder)
        # do align and cell find
        print('Align and Cell Find.\n')
        Okc = One_Key_Caiman(self.day_folder, self.runlist,boulder = self.boulder,in_server=self.in_server,align_base = self.align_base,max_shift=self.max_shift)
        Okc.Do_Caiman()
        # generate all condition dics.
        self.Cell_Cond_Dic = All_Cell_Condition_Generator(self.day_folder)
        # generate all t maps.
        if self.od_run != None:
            One_Key_T_Map(self.day_folder, self.od_run, self.od_type)
        if self.orien_run != None:
            One_Key_T_Map(self.day_folder, self.orien_run, self.orien_type)
        if self.color_run != None:
            One_Key_T_Map(self.day_folder, self.color_run, self.color_type)
        # generate tuning dictionary.
        print('Tuning Property calculating.')
        Tc = Tuning_Calculator(self.day_folder,
                               od_run = self.od_run,od_type = self.od_type,
                               orien_run = self.orien_run,orien_type = self.orien_type,
                               color_run = self.color_run,color_type = self.color_type)
        self.Cell_Tuning_Dic,self.Tuning_Property_Cells = Tc.Calculate_Tuning()
            
    
    
#%%
if __name__ == '__main__':
    
    day_folder = r'D:\ZR\_Temp_Data\210920_L76_2P'
    run_list = [1,2,3,6,7]
    pp = Preprocess_Pipeline(day_folder, run_list,boulder = (20,20,20,20),in_server = True,align_base='1-003')
    pp.Do_Preprocess()
    
    
    
    
    