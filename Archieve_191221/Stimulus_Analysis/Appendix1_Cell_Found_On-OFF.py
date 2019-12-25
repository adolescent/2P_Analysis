# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 12:47:25 2019

@author: ZR
This function is used to generate an On-Off Cell graph for stimulus situation, which might be more accurate in cell finding.

"""

#import General_Functions.my_tools as pp
import Step2_Cell_Found_In_A_Day as CF


root_folder = r'E:\ZR\Data_Temp\190412_L74_LM'
run_lists = ['004']
model_frame = r'Stim_Graphs_Morphology\On-Off.png'
find_type = 'On-Off'

cf = CF.Cell_Find_A_Day(32,root_folder,run_lists,1.5,model_frame,find_type)
cf.main()
cell_group_On_Off = cf.cell_group
#pp.save_variable(cell_group_On_Off,save_folder+r'\cell_group_On_Off.pkl')