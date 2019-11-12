# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 14:11:21 2019

@author: ZR
主程序，这里只要输入路径便可以一键式地得到减图和t图。

减法相关的参数需要预先包装，用工具Tool1即可。

如果是标准刺激，则可使用General_Functions.OI_Sub_Parameters内的预设参数进行计算。
"""

import Step1_Data_Produce as Data_Produce
import Step2_Sub_Map_Tool as Sub_Tool
import General_Functions.OI_Sub_Parameters.Standard_Stimulus as Standard_Stimulus


#%% User Input Here

blk_folder = r'E:\ZR\Data_Temp\191106_L69_OI\Run01_OD8'
Sub_parameter = Standard_Stimulus.G8_Parameters




#%%Usually NO Need to change files below.

# Step1 得到不同图的待减组合
CDG = Data_Produce.Condition_Data_Generate(blk_folder,sub_parameter)
CDG.Main()
Head_Property = CDG.Head_Property#这个是blk头文件的属性。这里注明方便接下来使用。
Produced_data = CDG.Produced_data
save_path = CDG.save_folder

# Step2,用以上的信息来生产减图。

SMP = Sub_Tool.Sub_Map_Produce(Head_Property,Produced_data,Sub_parameter,save_path)

