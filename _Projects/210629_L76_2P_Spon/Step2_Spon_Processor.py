# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 14:08:35 2021

@author: ZR
"""

import Spontaneous_Processor as SP


Sr = SP.Single_Run_Spontaneous_Processor(r'K:\Test_Data\2P\210629_L76_2P',
                                         spon_run = 'Run001')
PCA_Dic = Sr.Do_PCA(3700,9999)
Sr.Pairwise_Correlation_Plot(Sr.spon_cellname, 3700, 9999,'All_Before',cor_range = (-0.2,0.8))
Mu = SP.Multi_Run_Spontaneous_Processor(r'K:\Test_Data\2P\210629_L76_2P', 1.301)

#%% Evaluate cell fluctuation.
import Cell_2_DataFrame as C2D
import Cell_Train_Analyzer.Cell_Activity_Evaluator as CAE
All_Spon_Before = C2D.Multi_Run_Fvalue_Cat(r'K:\Test_Data\2P\210629_L76_2P', ['Run001','Run002','Run003'],rest_time = (600,600))
spike_count,Z_count = CAE.Spike_Count(All_Spon_Before)

#%% Get Tuing property of this day's run.
from Stimulus_Cell_Processor.Tuning_Property_Calculator import Tuning_Property_Calculator
import OS_Tools_Kit as ot

Tuning_0629 = Tuning_Property_Calculator(r'K:\Test_Data\2P\210629_L76_2P',
                                         Orien_para = ('Run004','G8_2P'),
                                         OD_para = ('Run006','OD_2P'),
                                         Hue_para = ('Run007','RGLum',False))


ot.Save_Variable(r'K:\Test_Data\2P\210629_L76_2P', 'All_Cell_Tuning', Tuning_0629,'.tuning')
