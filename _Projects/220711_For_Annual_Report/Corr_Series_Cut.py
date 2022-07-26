# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 17:24:36 2022

@author: ZR

"""

from Series_Analyzer.Preprocessor_Cai import Pre_Processor_Cai
import OS_Tools_Kit as ot

#%% Initailization
wp = r'D:\ZR\_Temp_Data\220711_temp'

series76 = Pre_Processor_Cai(r'D:\ZR\_Temp_Data\220630_L76_2P',start_frame=0,runname = 'Run001')
series85 = Pre_Processor_Cai(r'D:\ZR\_Temp_Data\220706_L85_LM',start_frame=0,runname = 'Run001')
series91 = Pre_Processor_Cai(r'D:\ZR\_Temp_Data\220420_L91',start_frame=0,runname = 'Run001')

ot.Save_Variable(wp, 'Series_76_Run1', series76)
ot.Save_Variable(wp, 'Series_85_Run1', series85)
ot.Save_Variable(wp, 'Series_91_Run1', series91)

