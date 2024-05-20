# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 19:44:49 2022

@author: ZR
"""

from Caiman_API.Precess_Pipeline import Preprocess_Pipeline
import OS_Tools_Kit as ot


day_folder = r'D:\ZR\_Temp_Data\220810_L85_2P'
pp = Preprocess_Pipeline(day_folder, [1,2,3,4,5,6,7])
pp.Do_Preprocess()
# wash cells.
boulder_new = [20,472,20,492]
acd_old = ot.Load_Variable(r'D:\ZR\_Temp_Data\220810_L85_2P\_CAIMAN\All_Series_Dic.pkl')
acn = list(acd_old.keys())
acn_new = {}
for i,cc in enumerate(acn):
    tc = acd_old[cc]
    if tc['Cell_Loc'][0]>boulder_new[0] and tc['Cell_Loc'][0]<boulder_new[1]:
        if tc['Cell_Loc'][1]>boulder_new[2] and tc['Cell_Loc'][1]<boulder_new[3]:
            acn_new[cc] = tc

ot.Save_Variable(r'D:\ZR\_Temp_Data\220810_L85_2P\_CAIMAN', 'All_Ceries_Dic_Washed', acn_new)


#%% Do pca.
from Series_Analyzer.Cell_Frame_PCA_Cai import One_Key_PCA
comp,info,weight = One_Key_PCA(day_folder,'Run001',tag = 'Spon_Before',start_frame = 5000)
comp_a,info_a,weight_a = One_Key_PCA(day_folder,'Run003',tag = 'Spon_After',start_frame = 0)


