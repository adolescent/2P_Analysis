# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 16:31:58 2021

@author: ZR
"""
from Stimulus_Cell_Processor.Tuning_Property_Calculator import Tuning_Property_Calculator
import OS_Tools_Kit as ot

day_folder = r'K:\Test_Data\2P\210721_L76_2P'
Tuning_0721 = Tuning_Property_Calculator(day_folder,
                                         Orien_para=('Run002','G16_2P'),
                                         OD_para=('Run006','OD_2P'),
                                         Hue_para=('Run007','HueNOrien4',{'Hue':['Red','Yellow','Green','Cyan','Blue','Purple','White']})
                                         )
ot.Save_Variable(r'K:\Test_Data\2P\210721_L76_2P','All_Cell_Tuning', Tuning_0721,'.tuning')
#%% Evaluate tuning basics
acn = list(Tuning_0721.keys())
LE_count = 0
RE_count = 0
for i,ccn in enumerate(acn):
    if Tuning_0721[ccn]['_OD_Preference'] == 'LE':
        LE_count +=1
    elif Tuning_0721[ccn]['_OD_Preference'] == 'RE':
        RE_count += 1