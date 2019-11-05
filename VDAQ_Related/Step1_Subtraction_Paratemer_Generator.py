# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 15:05:09 2019

@author: ZR

出于数据分析的方便考虑，在此提前把使用的参数在此包装起来，在step2里直接读取进去就可以进行参数读取了。

"""

import General_Functions.OS_Tools as OS_Tools

#Remember, Python list starts from Zero, so +1 when compared to matlab data.
OI_Parameters = {}
OI_Parameters['Ref_Frame']= [0,1]
OI_Parameters['Data_Frame'] = list(range(8,15))
OI_Parameters['Clip_std']= 1.5
OI_Parameters['Filter_Method'] = 'Gaussian'
OI_Parameters['HP_Filter_Parameter'] = [[2,2],2]#高通滤波的参数。对Gauss：两个维度分别是掩膜形状与标准差
OI_Parameters['LP_Filter_Parameter'] = [[50,50],50]#低通滤波的参数。对Gauss：两个维度分别是掩膜形状与标准差
#基本参数如上，接下来输入减图的名称和A/B Stim。一般来说，stm0的位置是放在最后的。

# 这一组参数是OD的
Sub_Sets = {}
Sub_Sets['L-R'] = [[1,2,3,4],[5,6,7,8]]
Sub_Sets['L-0'] = [[1,2,3,4],[9]]
Sub_Sets['R-0'] = [[5,6,7,8],[9]]
Sub_Sets['H-V'] = [[3,7],[1,5]]
Sub_Sets['O-A'] = [[4,8],[2,6]]
Sub_Sets['All-0'] = [[1,2,3,4,5,6,7,8],[9]]
OI_Parameters['All_Graph_Sets'] = Sub_Sets
OS_Tools.Save_And_Read.save_variable(OI_Parameters,'OD8_Parameters.pkl')


# =============================================================================
# #这一组参数是RGLum4的
# Sub_Sets = {}
# Sub_Sets['RG-Lum']=[[1,2],[3,4]]
# Sub_Sets['O-A'] = [[1,3],[2,4]]
# Sub_Sets['RG-0'] = [[1,2],[5]]
# Sub_Sets['Lum-0'] = [[3,4],[5]]
# Sub_Sets['All-0'] = [[1,2,3,4],[5]]
# OI_Parameters['All_Graph_Sets'] = Sub_Sets
# OS_Tools.Save_And_Read.save_variable(OI_Parameters,'RGLum4_Parameters.pkl')
# =============================================================================

# =============================================================================
# # 这一组参数是G8的
# Sub_Sets = {}
# Sub_Sets['H-V'] = [[3,7],[1,5]]
# Sub_Sets['O-A'] = [[4,8],[2,6]]
# Sub_Sets['Ori0-0'] =[[3,7],[9]]
# Sub_Sets['Ori45-0'] = [[4,8],[9]] 
# Sub_Sets['Ori90-0'] = [[1,5],[9]] 
# Sub_Sets['Ori135-0'] = [[2,6],[9]] 
# Sub_Sets['HV-OA']= [[1,3,5,7],[2,4,6,8]]
# Sub_Sets['DirL-R'] = [[4,5,6],[2,1,8]]
# Sub_Sets['DirU-D'] = [[2,3,4],[6,7,8]]
# Sub_Sets['All-0'] = [[1,2,3,4,5,6,7,8],[9]]
# OI_Parameters['All_Graph_Sets'] = Sub_Sets
# OS_Tools.Save_And_Read.save_variable(OI_Parameters,'G8_Parameters.pkl')
# =============================================================================

#%%
