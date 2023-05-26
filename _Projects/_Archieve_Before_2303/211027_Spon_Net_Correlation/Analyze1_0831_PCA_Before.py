# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 15:30:15 2021

@author: ZR
"""

from Series_Analyzer.Cell_Frame_PCA import Do_PCA
from Series_Analyzer.Cell_Frame_PCA import Compoment_Visualize
import OS_Tools_Kit as ot
import numpy as np
import List_Operation_Kit as lt

#%% Load Variables
base_folder = r'G:\_Processed_Results\211027_Spon_Net_Correlation\Origin_Data'
before_frame = ot.Load_Variable(base_folder,'0831_Before.pkl')
all_cell_dic = ot.Load_Variable(base_folder,'L76_210831A_All_Cells.ac')
#%% Get all stim t maps.
G16_t_info = ot.Load_Variable(r'G:\Test_Data\2P\210831_L76_2P\_All_Results\G16_2P_t_Maps\All_T_Info.pkl')['cell_info']
OD_t_info = ot.Load_Variable(r'G:\Test_Data\2P\210831_L76_2P\_All_Results\OD_2P_t_Maps\All_T_Info.pkl')['cell_info']
Hue_t_info = ot.Load_Variable(r'G:\Test_Data\2P\210831_L76_2P\_All_Results\HueNOrien4_t_Maps\All_T_Info.pkl')['cell_info']
G16_keys = list(G16_t_info.keys())[1:14]
all_t_info = dict((k, G16_t_info[k]) for k in (G16_keys))
all_t_info['L-0'] = OD_t_info['L-0']
all_t_info['R-0'] = OD_t_info['R-0']
all_t_info['OD'] = OD_t_info['OD']
Hue_keys = np.array(list(Hue_t_info.keys()))[[4,6,8,10,12,14]].tolist()
for i,c_key in enumerate(Hue_keys):
    all_t_info[c_key] = Hue_t_info[c_key]
ot.Save_Variable(r'G:\Test_Data\2P\210831_L76_2P\_All_Results', 'All_Stim_Graph', all_t_info)




#%% Analyze
output_folder = r'G:\_Processed_Results\211027_Spon_Net_Correlation\PCA_Results'
components,PCA_info,fitted_weights = Do_PCA(before_frame)
all_PC_graphs = Compoment_Visualize(components, all_cell_dic, output_folder)
ot.Save_Variable(output_folder, 'PC_Components', components)
ot.Save_Variable(output_folder, 'PC_info', PCA_info)
ot.Save_Variable(output_folder, 'fitted_weights', fitted_weights)
from Stimulus_Cell_Processor.Map_Tuning_Calculator import PC_Tuning_Calculation
PC_comp_tunings = PC_Tuning_Calculation(components, r'G:\Test_Data\2P\210831_L76_2P')
ot.Save_Variable(output_folder, 'PCA_Tunings', PC_comp_tunings)



