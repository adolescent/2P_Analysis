# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 15:05:09 2019

@author: ZR

出于数据分析的方便考虑，在此提前把使用的参数在此包装起来，在step2里直接读取进去就可以进行参数读取了。

"""

import General_Functions.OS_Tools as OS_Tools
import numpy as np

#Remember, Python list starts from Zero, so +1 when compared to matlab data.
OI_Parameters = {}
OI_Parameters['Ref_Frame']= [0,1]
OI_Parameters['Data_Frame'] = list(range(8,15))
OI_Parameters['Clip_std']= 1.5
OI_Parameters['Filter_Method'] = 'Gaussian'
OI_Parameters['HP_Filter_Parameter'] = [[2,2],2]#高通滤波的参数。对Gauss：两个维度分别是掩膜形状与标准差
OI_Parameters['LP_Filter_Parameter'] = [[50,50],50]#低通滤波的参数。对Gauss：两个维度分别是掩膜形状与标准差
#基本参数如上，接下来输入减图的名称和A/B Stim
Sub_Sets = {}
Sub_Sets['H-V'] = [[],[]]