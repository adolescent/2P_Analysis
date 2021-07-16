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
