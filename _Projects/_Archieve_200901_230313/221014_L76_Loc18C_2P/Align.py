# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 20:24:23 2022

@author: ZR
"""

from My_Wheels.Caiman_API.Precess_Pipeline import Preprocess_Pipeline
import OS_Tools_Kit as ot

wp = r'D:\ZR\_Temp_Data\221014_L76_2P'
pp = Preprocess_Pipeline(wp, [1,2,6,8,9],boulder = (20,20,20,30),align_base='1-002',
                         od_run = 'Run009',orien_run='Run006',color_run = 'Run008')

pp.Do_Preprocess()
acd = ot.Load_Variable(r'D:\ZR\_Temp_Data\221014_L76_2P\_CAIMAN','All_Series_Dic.pkl')

#%% Do PCA
from Series_Analyzer.Cell_Frame_PCA_Cai import One_Key_PCA,Do_PCA,Comp_Visualize_Cai
from Series_Analyzer.Preprocessor_Cai import Pre_Processor_Cai
comp,info,weight = One_Key_PCA(wp, 'Run001',tag = 'Spon_Before',start_frame = 2500)
comp_a,info_a,weight_a = One_Key_PCA(wp, 'Run002',tag = 'Spon_After',start_frame = 0)
Run02_Frame = Pre_Processor_Cai(wp,'Run002')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
spon1 = Run02_Frame.loc[:,1200:4800]
spon2 = Run02_Frame.loc[:,6200:9500]
spon3 = Run02_Frame.loc[:,10900:14100]
spon4 = Run02_Frame.loc[:,15300:]

all_spon = pd.concat([spon1,spon2,spon3,spon4],axis = 1)
comp,info,weight = Do_PCA(all_spon)
PC_name = comp.columns
for i,c_PC in enumerate(PC_name):
    c_comp = comp.loc[:,c_PC]
    c_graph = Comp_Visualize_Cai(c_comp,acd,show = False)
    fig = plt.figure(figsize = (15,15))
    plt.title(c_PC,fontsize=36)
    fig = sns.heatmap(c_graph,square=True,yticklabels=False,xticklabels=False,center = 0)
    fig.figure.savefig(ot.join(r'D:\ZR\_Temp_Data\221014_L76_2P\_CAIMAN\Run02_Spon_Only',c_PC)+'.png')
    plt.clf()
    plt.close()
    