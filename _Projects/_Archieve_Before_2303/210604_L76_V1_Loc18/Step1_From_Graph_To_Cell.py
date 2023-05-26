# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 21:40:54 2021

@author: ZR
"""

from Standard_Aligner import Standard_Aligner
SA = Standard_Aligner(r'K:\Test_Data\2P\210604_L76_2P', list(range(1,17)))
SA.One_Key_Aligner()
#%%
from Stim_Frame_Align import One_Key_Stim_Align
One_Key_Stim_Align(r'K:\Test_Data\2P\210604_L76_2P\210604_stimuli')




#%%
from Standard_Stim_Processor import One_Key_Frame_Graphs
from Standard_Parameters import Sub_Graph_Dics
OD_Para = Sub_Graph_Dics.Sub_Dic_Generator('OD_2P')
One_Key_Frame_Graphs(r'K:\Test_Data\2P\210604_L76_2P\1-008', OD_Para)
G16_Para = Sub_Graph_Dics.Sub_Dic_Generator('G16_2P')
One_Key_Frame_Graphs(r'K:\Test_Data\2P\210604_L76_2P\1-014', G16_Para)
One_Key_Frame_Graphs(r'K:\Test_Data\2P\210604_L76_2P\1-016', G16_Para)
Hue_Para = Sub_Graph_Dics.Sub_Dic_Generator('HueNOrien4',{'Hue':['Red','Yellow','Green','Cyan','Blue','Purple','White']})
One_Key_Frame_Graphs(r'K:\Test_Data\2P\210604_L76_2P\1-015', Hue_Para)

