# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 15:06:18 2021

@author: ZR
"""
import Graph_Operation_Kit as gt
import OS_Tools_Kit as ot
import numpy as np
from Cell_Processor import Cell_Processor
import random
import Statistic_Tools as st
import matplotlib.pyplot as plt
from Spontaneous_Processor import Spontaneous_Processor
from Cross_Day_Cell_Layout import Cross_Day_Cell_Layout
from Spontaneous_Processor import Cross_Run_Pair_Correlation
import seaborn as sns
import pandas as pd

work_path = r'D:\ZR\_MyCodes\2P_Analysis\_Projects\210616_Annual_Report'
#%% Graph1, generate average graph of different run.
# Use G8 response as graph base.
graph_names_0604 = ot.Get_File_Name(r'K:\Test_Data\2P\210604_L76_2P\1-016\Results\Final_Aligned_Frames')
avr_0604 = gt.Average_From_File(graph_names_0604)
graph_names_0123 = ot.Get_File_Name(r'K:\Test_Data\2P\210123_L76_2P\1-011\Results\Aligned_Frames')
avr_0123 = gt.Average_From_File(graph_names_0123)
clipped_0604 = np.clip((avr_0604.astype('f8'))*30,0,65535).astype('u2')
clipped_0123 = np.clip((avr_0123.astype('f8'))*30,0,65535).astype('u2')
gt.Show_Graph(clipped_0604, 'Average_0604',work_path)
gt.Show_Graph(clipped_0123, 'Average_0123',work_path)
#%% Graph2, get cell layout of 210401 and 210413
CP_0401 = Cell_Processor(r'K:\Test_Data\2P\210401_L76_2P')
CP_0413 = Cell_Processor(r'K:\Test_Data\2P\210413_L76_2P')
all_cell_name_0401 = CP_0401.all_cell_names
all_cell_name_0413 = CP_0413.all_cell_names
h = Cross_Day_Cell_Layout(r'K:\Test_Data\2P\210401_L76_2P',r'K:\Test_Data\2P\210413_L76_2P', all_cell_name_0401, all_cell_name_0413)
#%% Analyze spon series of 0413.
SP = Spontaneous_Processor(r'K:\Test_Data\2P\210413_L76_2P',spon_run = 'Run001')
pc_dic_0_10 = SP.Do_PCA(0,600)
pc_dic_10_20 = SP.Do_PCA(600,1200)
pc_dic_20_30 = SP.Do_PCA(1200,1800)
pc_dic_30_40 = SP.Do_PCA(1800,2400)
pc_dic_0_45_all = SP.Do_PCA(0,2770)
pc_dic_stim_10 = SP.Do_PCA(2800,3400)
test_SP = Spontaneous_Processor(r'K:\Test_Data\2P\210413_L76_2P',spon_run = 'Run007')
test_PCA = test_SP.Do_PCA()
after_SP = Spontaneous_Processor(r'K:\Test_Data\2P\210413_L76_2P',spon_run = 'Run009')
after_pc_all = after_SP.Do_PCA()
after_pc_0_10 = after_SP.Do_PCA(0,600)
after_pc_50_60 = after_SP.Do_PCA(3000,3600)
#%% get video show
from My_Wheels.Video_Writer import Video_From_File
Video_From_File(r'K:\Test_Data\2P\210413_L76_2P\1-001\Results\Final_Aligned_Frames',plot_range = (0,2700))
Video_From_File(r'K:\Test_Data\2P\210413_L76_2P\1-009\Results\Final_Aligned_Frames')
#%% Get different tuning index and do cross correlation.
from My_Wheels.Cell_Processor import Cell_Processor
CP = Cell_Processor(r'K:\Test_Data\2P\210413_L76_2P')
OD_index = CP.Index_Calculator_Core('Run007', [1,3,5,7], [2,4,6,8])
OD_T_Data = CP.T_Map_Plot_Core('Run007', [1,3,5,7], [2,4,6,8])
all_cell_name = list(OD_index.keys())
# get cell name list of special indexes.
LE_cells = []
RE_cells = []
for i in range(len(all_cell_name)):
    c_name = all_cell_name[i]
    if OD_index[c_name] != None:
        if OD_index[c_name]['Tuning_Index']>0.5 and OD_index[c_name]['p_value']<0.05:
            LE_cells.append(c_name)
        if OD_index[c_name]['Tuning_Index']<-0.5 and OD_index[c_name]['p_value']<0.05:
            RE_cells.append(c_name)
SP_1 = Spontaneous_Processor(r'K:\Test_Data\2P\210413_L76_2P',spon_run = 'Run001')
LE_ac = SP_1.Pairwise_Correlation_Core(LE_cells, 0, 2700)
LE_rand_cells = random.sample(all_cell_name, 108)
LE_rand_ac_1 = SP_1.Pairwise_Correlation_Core(LE_rand_cells, 0, 2700)
RE_ac = SP_1.Pairwise_Correlation_Core(RE_cells, 0, 2700)
RE_rand_cells = random.sample(all_cell_name, 83)
RE_rand_ac_1 = SP_1.Pairwise_Correlation_Core(RE_rand_cells, 0, 2700)
All_ac = SP_1.Pairwise_Correlation_Core(all_cell_name, 0, 2700)
LE_rand_ac_2 = random.sample(All_ac, 5778)
RE_rand_ac_2 = random.sample(All_ac, 3403)


fig,ax = plt.subplots(figsize = (12,8))
bins = np.linspace(-0.2, 0.2, 200)
ax.hist(RE_rand_ac_1,bins,label ='Random')
ax.hist(RE_ac,bins,label = 'RE')
ax.legend(prop={'size': 20})
t,p,_ = st.T_Test_Ind(RE_ac, RE_rand_ac_1)
ax.annotate('t ='+str(round(t,3)),xycoords = 'axes fraction',xy = (0.9,0.7))
ax.annotate('p ='+str(round(p,5)),xycoords = 'axes fraction',xy = (0.9,0.65))
fig.savefig(work_path+r'\Hist.png',dpi=180)

# Run09 Pair Correlation
SP_2 = Spontaneous_Processor(r'K:\Test_Data\2P\210413_L76_2P',spon_run = 'Run009')
LE_ac_after = SP_2.Pairwise_Correlation_Core(LE_cells, 0, 2700)
LE_rand_cells = random.sample(all_cell_name, 108)
LE_rand_ac_1_after = SP_1.Pairwise_Correlation_Core(LE_rand_cells, 0, 2700)
RE_ac_after = SP_2.Pairwise_Correlation_Core(RE_cells, 0, 2700)
RE_rand_cells = random.sample(all_cell_name, 83)
RE_rand_ac_1_after = SP_1.Pairwise_Correlation_Core(RE_rand_cells, 0, 2700)

fig,ax = plt.subplots(figsize = (12,8))
bins = np.linspace(-0.2, 0.8, 200)
ax.hist(RE_ac,bins,label ='RE_BEFORE')
ax.hist(RE_ac_after,bins,label = 'RE_AFTER')
ax.legend(prop={'size': 20})
t,p,_ = st.T_Test_Ind(RE_ac, RE_ac_after)
ax.annotate('t ='+str(round(t,3)),xycoords = 'axes fraction',xy = (0.9,0.7))
ax.annotate('p ='+str(round(p,5)),xycoords = 'axes fraction',xy = (0.9,0.65))
fig.savefig(work_path+r'\Hist.png',dpi=180)
# Stim ON ac
SP_1.Pairwise_Correlation_Plot(LE_cells,2800,3400,'LE',cor_range = (-0.2,0.5))
SP_1.Pairwise_Correlation_Plot(RE_cells,2800,3400,'RE',cor_range = (-0.2,0.5))

# Compare Run01 vs Run09 Global correlation
run01_spon_cells = SP_1.spon_cellname
run09_spon_cells = SP_2.spon_cellname
All_ac_Before = SP_1.Pairwise_Correlation_Core(run01_spon_cells, 0, 2700)
All_ac_After = SP_2.Pairwise_Correlation_Core(run09_spon_cells, 0, 2700)
    
fig,ax = plt.subplots(figsize = (12,8))
bins = np.linspace(-0.2, 0.5, 200)
ax.hist(All_ac_Before,bins,label ='All_BEFORE')
ax.hist(All_ac_After,bins,label = 'All_AFTER')
ax.legend(prop={'size': 20})
t,p,_ = st.T_Test_Ind(All_ac_After, All_ac_Before)
ax.annotate('t ='+str(round(t,3)),xycoords = 'axes fraction',xy = (0.9,0.7))
ax.annotate('p ='+str(round(p,5)),xycoords = 'axes fraction',xy = (0.9,0.65))
fig.savefig(work_path+r'\Hist.png',dpi=180)

# Compare Run09 10min before and 10 min after.
Run09_All_0_10 = SP_2.Pairwise_Correlation_Core(all_cell_name, 0, 600)
Run09_All_50_60 = SP_2.Pairwise_Correlation_Core(all_cell_name, 3000,3600)
fig,ax = plt.subplots(figsize = (12,8))
bins = np.linspace(-0.2, 0.7, 200)
ax.hist(Run09_All_50_60,bins,label ='50-60 min')
ax.hist(Run09_All_0_10,bins,label = '0-10 min')
ax.legend(prop={'size': 20})
t,p,_ = st.T_Test_Ind(Run09_All_50_60, Run09_All_0_10)
ax.annotate('t ='+str(round(t,3)),xycoords = 'axes fraction',xy = (0.9,0.7))
ax.annotate('p ='+str(round(p,5)),xycoords = 'axes fraction',xy = (0.9,0.65))
fig.savefig(work_path+r'\Hist.png',dpi=180)
#%% Difference of HV cells
from My_Wheels.Cell_Processor import Cell_Processor
CP = Cell_Processor(r'K:\Test_Data\2P\210413_L76_2P')
HV_index = CP.Index_Calculator_Core('Run013', [1,9], [5,13])
HV_T_Data = CP.T_Map_Plot_Core('Run013', [1,9], [5,13])
all_cell_name = list(HV_index.keys())
# get cell name list of special indexes.
H_cells = []
V_cells = []
all_index = []
for i in range(len(all_cell_name)):
    c_name = all_cell_name[i]
    if HV_index[c_name] != None:
        all_index.append(HV_index[c_name]['Tuning_Index'])
        if HV_index[c_name]['Tuning_Index']>0.5 and HV_index[c_name]['p_value']<0.05:
            H_cells.append(c_name)
        if HV_index[c_name]['Tuning_Index']<-0.5 and HV_index[c_name]['p_value']<0.05:
            V_cells.append(c_name)
SP_1.Pairwise_Correlation_Plot(H_cells,0,2700,'Horizontal',cor_range = (-0.2,0.2))
SP_1.Pairwise_Correlation_Plot(V_cells,0,2700,'Vertical',cor_range = (-0.2,0.2))
SP_2.Pairwise_Correlation_Plot(H_cells,0,2700,'Horizontal',cor_range = (-0.2,0.7))
SP_2.Pairwise_Correlation_Plot(V_cells,0,2700,'Vertical',cor_range = (-0.2,0.7))
 
# Compare LE network with Horizontal network.
selected_V_cell = random.sample(V_cells,108)
selected_random_cell = random.sample(all_cell_name, 108)

random_data = SP_2.Pairwise_Correlation_Core(selected_random_cell, 0, 2700)
V_data = SP_2.Pairwise_Correlation_Core(selected_V_cell, 0, 2700)
LE_data = SP_2.Pairwise_Correlation_Core(LE_cells, 0, 2700)
fig,ax = plt.subplots(figsize = (12,8))
bins = np.linspace(-0.1, 0.7, 200)
ax.hist(random_data,bins,label ='Random',alpha = 0.75)
ax.hist(LE_data,bins,label = 'Left Eye',alpha = 0.75)
ax.hist(V_data,bins,label = 'Vertical',alpha = 0.75)
ax.legend(prop={'size': 20})
t,p,_ = st.T_Test_Ind(V_data, LE_data)
ax.annotate('t ='+str(round(t,3)),xycoords = 'axes fraction',xy = (0.9,0.7))
ax.annotate('p ='+str(round(p,5)),xycoords = 'axes fraction',xy = (0.9,0.65))
fig.savefig(work_path+r'\Hist.png',dpi=180)
#%% Difference of Color cells
from My_Wheels.Cell_Processor import Cell_Processor
CP = Cell_Processor(r'K:\Test_Data\2P\210413_L76_2P')
RB_index = CP.Index_Calculator_Core('Run014', [5,16,27,38],[9,20,31,42])
RB_T_Data = CP.T_Map_Plot_Core('Run014', [5,16,27,38], [9,20,31,42])
all_cell_name = list(RB_index.keys())
# get cell name list of special indexes.
Red_cells = []
Blue_cells = []
all_index = []
for i in range(len(all_cell_name)):
    c_name = all_cell_name[i]
    if RB_index[c_name] != None:
        all_index.append(RB_index[c_name]['t_value'])
        if RB_index[c_name]['Tuning_Index']>0.5 and RB_index[c_name]['p_value']<0.05:
            Red_cells.append(c_name)
        if RB_index[c_name]['Tuning_Index']<-0.5 and RB_index[c_name]['p_value']<0.05:
            Blue_cells.append(c_name)
            
SP_1.Pairwise_Correlation_Plot(Red_cells,0,2700,'Red',cor_range = (-0.2,0.2))
SP_1.Pairwise_Correlation_Plot(Blue_cells,0,2700,'Blue',cor_range = (-0.2,0.2))
SP_2.Pairwise_Correlation_Plot(Red_cells,0,2700,'Red',cor_range = (-0.2,0.7))
SP_2.Pairwise_Correlation_Plot(Blue_cells,0,2700,'Blue',cor_range = (-0.2,0.7))
# Then compare red and blue with orientation.
selected_blue = random.sample(Blue_cells, 49)
selected_vertical = random.sample(V_cells, 49)

red_data = SP_2.Pairwise_Correlation_Core(Red_cells, 0, 2700)
blue_data = SP_2.Pairwise_Correlation_Core(selected_blue, 0, 2700)
vertical_data = SP_2.Pairwise_Correlation_Core(selected_vertical, 0, 2700)

fig,ax = plt.subplots(figsize = (12,8))
bins = np.linspace(-0.2, 0.7, 200)
ax.hist(red_data,bins,label ='Red',alpha = 0.75)
ax.hist(vertical_data,bins,label = 'Vertical',alpha = 0.75)
ax.hist(blue_data,bins,label = 'Blue',alpha = 0.75)
ax.legend(prop={'size': 20})
t,p,_ = st.T_Test_Ind(red_data, blue_data)
ax.annotate('t ='+str(round(t,3)),xycoords = 'axes fraction',xy = (0.9,0.7))
ax.annotate('p ='+str(round(p,5)),xycoords = 'axes fraction',xy = (0.9,0.65))
fig.savefig(work_path+r'\Hist.png',dpi=180)


#%% Seed point method.
from My_Wheels.Cell_Processor import Cell_Processor
CP = Cell_Processor(r'K:\Test_Data\2P\210413_L76_2P')
all_cell_name = CP.all_cell_names
RB_index = CP.Index_Calculator_Core('Run014', [5,16,27,38],[9,20,31,42])
HV_index = CP.Index_Calculator_Core('Run013', [1,9], [5,13])
OD_index = CP.Index_Calculator_Core('Run007', [1,3,5,7], [2,4,6,8])
All_Seed_Property = {}
for i in range(len(all_cell_name)):
    c_name = all_cell_name[i]
    c_hv = HV_index[c_name]
    All_Seed_Property[c_name] = []
    if c_hv != None:
        if c_hv['Tuning_Index']>0.5 and c_hv['p_value']<0.05:
            All_Seed_Property[c_name].append('Horizontal')
        if c_hv['Tuning_Index']<-0.5 and c_hv['p_value']<0.05:
            All_Seed_Property[c_name].append('Vertical')
    c_rb = RB_index[c_name]
    if c_rb != None:
        if c_rb['Tuning_Index']>0.5 and c_rb['p_value']<0.05:
            All_Seed_Property[c_name].append('Red')
        if c_rb['Tuning_Index']<-0.5 and c_rb['p_value']<0.05:
            All_Seed_Property[c_name].append('Blue')
    c_od = OD_index[c_name]
    if c_od != None:
        if c_od['Tuning_Index']>0.5 and c_od['p_value']<0.05:
            All_Seed_Property[c_name].append('LE')
        if c_od['Tuning_Index']<-0.5 and c_od['p_value']<0.05:
            All_Seed_Property[c_name].append('RE')
SP_Before = Spontaneous_Processor(r'K:\Test_Data\2P\210413_L76_2P','Run001')
SP_After = Spontaneous_Processor(r'K:\Test_Data\2P\210413_L76_2P','Run009')
SP_Test = Spontaneous_Processor(r'K:\Test_Data\2P\210413_L76_2P','Run013')
for i in range(len(all_cell_name)):
    c_sp = all_cell_name[i]
    SP_Before.Seed_Point_Correlation_Map(c_sp,0,2700,seed_brightnes = 0.15)
    
    
#%% Compare V1 vs V2 cell sizes.
V1_Cells = ot.Load_Variable(r'K:\Test_Data\2P\210413_L76_2P\L76_210413A_All_Cells.ac')
V2_Cells = ot.Load_Variable(r'K:\Test_Data\2P\210504_L76_2P\L76_210504A_All_Cells.ac')
V1_cn = list(V1_Cells.keys())
V2_cn = list(V2_Cells.keys())
V1_cell_size = []
V2_cell_size = []
for i in range(len(V1_cn)):
    c_c = V1_Cells[V1_cn[i]]
    V1_cell_size.append(c_c['Cell_Area'])
for i in range(len(V2_cn)):
    c_c = V2_Cells[V2_cn[i]]
    V2_cell_size.append(c_c['Cell_Area'])
    
    
fig,ax = plt.subplots(figsize = (12,8))
bins = np.linspace(0,200,20)
ax.hist(V2_cell_size,bins,label ='V2 Size',alpha=0.8)
ax.hist(V1_cell_size,bins,label ='V1 Size',alpha=0.8)
ax.legend(prop={'size': 20})
t,p,_ = st.T_Test_Ind(V2_cell_size,V1_cell_size)
ax.annotate('t ='+str(round(t,3)),xycoords = 'axes fraction',xy = (0.9,0.7))
ax.annotate('p ='+str(round(p,10)),xycoords = 'axes fraction',xy = (0.9,0.65))
fig.savefig(r'Size_Compare.png',dpi=180)
#%% Do Pairwise correlation for V2 cells, orientation 0 and 90
CP = Cell_Processor(r'K:\Test_Data\2P\210504_L76_2P')
HV_index = CP.Index_Calculator_Core('Run013',[1,9],[5,13])
all_cell_name = list(HV_index.keys())
# get cell name list of special indexes.
Horiziontal_cells = []
Vertical_cells = []
for i in range(len(all_cell_name)):
    c_name = all_cell_name[i]
    if HV_index[c_name] != None:
        if HV_index[c_name]['Tuning_Index']>0.5 and HV_index[c_name]['p_value']<0.05:
            Horiziontal_cells.append(c_name)
        if HV_index[c_name]['Tuning_Index']<-0.5 and HV_index[c_name]['p_value']<0.05:
            Vertical_cells.append(c_name)
Spon_Before = Spontaneous_Processor(r'K:\Test_Data\2P\210504_L76_2P','Run001')
before_corr = Spon_Before.Pairwise_Correlation_Plot(Vertical_cells,0,1800,'Vertical',(-0.2,0.6))
Spon_After = Spontaneous_Processor(r'K:\Test_Data\2P\210504_L76_2P','Run014')
after_corr = Spon_After.Pairwise_Correlation_Plot(Vertical_cells,0,1800,'Vertical',(-0.2,0.6))
Ver_Before_Corr,Ver_After_Corr = Cross_Run_Pair_Correlation(r'K:\Test_Data\2P\210504_L76_2P', 
                                                            Vertical_cells,'Run001','Run014',0,1800,0,1800,
                                                            'Vertical_Before','Vertical_After')
All_Before_Corr,All_After_Corr = Cross_Run_Pair_Correlation(r'K:\Test_Data\2P\210504_L76_2P', 
                                                            all_cell_name,'Run001','Run014',0,1800,0,1800,
                                                            'All_Before','All_After')


# Get data frame here.
all_corr_dic = {}
all_corr_dic['Before'] = []
all_corr_dic['After'] = []
all_corr_dic['Origin'] = []
for i in range(len(Ver_Before_Corr)):
    all_corr_dic['Before'].append(Ver_Before_Corr[i])
    all_corr_dic['After'].append(Ver_After_Corr[i])
    all_corr_dic['Origin'].append('Vertical')
for i in range(len(All_Before_Corr)):
    all_corr_dic['Before'].append(All_Before_Corr[i])
    all_corr_dic['After'].append(All_After_Corr[i])
    all_corr_dic['Origin'].append('All')    
Correlation_Frame_V2 = pd.DataFrame(all_corr_dic)
selected_Frame_V2 = Correlation_Frame_V2.groupby('Origin').sample(5670)    
fig = sns.jointplot(data=selected_Frame_V2, x="Before", y="After", hue="Origin",kind="kde",height = 8,fill =True,xlim = (-0.2,0.5),ylim = (-0.2,0.5),thres = 0.1,alpha = 0.6)
fig.ax_joint.plot([-0.2,0.5],[-0.2,0.5], 'r--')
fig.savefig(work_path+r'\Kernel.png',dpi = 180)

#%% Add jointplot for V1 results...
LE_Before,LE_After = Cross_Run_Pair_Correlation(r'K:\Test_Data\2P\210413_L76_2P', 
                                                LE_cells,'Run001','Run009',0,2700,0,2700,
                                                'LE_Before','LE_After',cor_range = (-0.2,0.7))
RE_Before,RE_After = Cross_Run_Pair_Correlation(r'K:\Test_Data\2P\210413_L76_2P', 
                                                RE_cells,'Run001','Run009',0,2700,0,2700,
                                                'RE_Before','RE_After',cor_range = (-0.2,0.7))
All_V1_Before,All_V1_After = Cross_Run_Pair_Correlation(r'K:\Test_Data\2P\210413_L76_2P', 
                                                        all_cell_name,'Run001','Run009',0,2700,0,2700,
                                                        'All_Before','All_After',cor_range = (-0.2,0.7))
All_V1_After_0_10,All_V1_After_50_60 = Cross_Run_Pair_Correlation(r'K:\Test_Data\2P\210413_L76_2P', 
                                                                  all_cell_name,'Run009','Run009',0,600,3000,3600,
                                                                  'All 0-10 min','All 50-60 min',cor_range = (-0.2,0.7))
V_Before_V1,V_After_V1 = Cross_Run_Pair_Correlation(r'K:\Test_Data\2P\210413_L76_2P', 
                                                    V_cells,'Run001','Run009',0,2700,0,2700,
                                                    'Vertical_Before','Vertical_After',cor_range = (-0.2,0.7))
Blue_Before_V1,Blue_After_V1 = Cross_Run_Pair_Correlation(r'K:\Test_Data\2P\210413_L76_2P', 
                                                          Blue_cells,'Run001','Run009',0,2700,0,2700,
                                                          'Vertical_Before','Vertical_After',cor_range = (-0.2,0.7))
Red_Before_V1,Red_After_V1 = Cross_Run_Pair_Correlation(r'K:\Test_Data\2P\210413_L76_2P', 
                                                        Red_cells,'Run001','Run009',0,2700,0,2700,
                                                        'Vertical_Before','Vertical_After',cor_range = (-0.2,0.7))
# Get data frame here.
all_corr_dic_V1 = {}
all_corr_dic_V1['Before'] = []
all_corr_dic_V1['After'] = []
all_corr_dic_V1['Origin'] = []
for i in range(len(LE_Before)):
    all_corr_dic_V1['Before'].append(LE_Before[i])
    all_corr_dic_V1['After'].append(LE_After[i])
    all_corr_dic_V1['Origin'].append('LE')
for i in range(len(V_Before_V1)):
    all_corr_dic_V1['Before'].append(V_Before_V1[i])
    all_corr_dic_V1['After'].append(V_After_V1[i])
    all_corr_dic_V1['Origin'].append('Vertical')    
for i in range(len(Red_Before_V1)):
    all_corr_dic_V1['Before'].append(Red_Before_V1[i])
    all_corr_dic_V1['After'].append(Red_After_V1[i])
    all_corr_dic_V1['Origin'].append('Red')  
for i in range(len(Blue_Before_V1)):
    all_corr_dic_V1['Before'].append(Blue_Before_V1[i])
    all_corr_dic_V1['After'].append(Blue_After_V1[i])
    all_corr_dic_V1['Origin'].append('Blue')  
for i in range(len(All_V1_Before)):
    all_corr_dic_V1['Before'].append(All_V1_Before[i])
    all_corr_dic_V1['After'].append(All_V1_After[i])
    all_corr_dic_V1['Origin'].append('All')
    
Correlation_Frame_V1 = pd.DataFrame(all_corr_dic_V1)
Selected_Frame_V1 = Correlation_Frame_V1.groupby('Origin').sample(1100)
fig = sns.jointplot(data=Selected_Frame_V1, x="Before", y="After",kind = 'kde', hue="Origin",height = 8,xlim = (-0.2,0.2),ylim = (-0.2,0.7),levels = [0.1,0.9], thresh=.1,alpha = 0.7)
fig.ax_joint.plot([-0.2,0.5],[-0.2,0.5], ls = '--',color = 'gray')
fig.savefig(work_path+r'\Kernel.png',dpi = 180)
#%% Compare V1 and V2 Glocal
All_Frame_V1 = Correlation_Frame_V1.loc[Correlation_Frame_V1['Origin'] == 'All']
All_Frame_V2 = Correlation_Frame_V2.loc[Correlation_Frame_V2['Origin'] == 'All']
All_Frame_V1['Area'] = 'V1'
All_Frame_V2['Area'] = 'V2'
Area_Compare_Frame = pd.concat([All_Frame_V1,All_Frame_V2])
Selected_Frames = Area_Compare_Frame.groupby('Area').sample(19000)
fig = sns.jointplot(data=Selected_Frames, x="Before", y="After",kind = 'kde', hue="Area",height = 8,xlim = (-0.2,0.5),ylim = (-0.2,0.5),fill = True, thresh=.1,alpha = 0.7)
fig = sns.jointplot(data=Selected_Frames, x="Before", y="After",hue="Area",height = 8,xlim = (-0.2,0.6),ylim = (-0.2,0.6),s = 4,alpha = 0.7)
fig.ax_joint.plot([-0.2,0.6],[-0.2,0.6], ls = '--',color = 'gray')
fig.savefig(work_path+r'\Kernel.png',dpi = 180)

#%% Another set of data(210423-V1), just in case the result is fake..

CP_Repeat = Cell_Processor(r'K:\Test_Data\2P\210423_L76_2P')
ac_repeat = CP_Repeat.all_cell_names
All_Before_Corr,All_After_Corr = Cross_Run_Pair_Correlation(r'K:\Test_Data\2P\210423_L76_2P', 
                                                            ac_repeat,'Run001','Run015',0,1800,0,1800,
                                                            'All_Before','All_After')
CP_Repeat_V2 = Cell_Processor(r'K:\Test_Data\2P\210514_L76_2P')
ac_repeat_V2 = CP_Repeat_V2.all_cell_names
All_Before_Corr_V2,All_After_Corr_V2 = Cross_Run_Pair_Correlation(r'K:\Test_Data\2P\210514_L76_2P', 
                                                                  ac_repeat_V2,'Run001','Run015',0,1800,0,1800,
                                                                  'All_Before','All_After')
Repeat_Corr = {}
Repeat_Corr['Before'] = []
Repeat_Corr['After'] = []
Repeat_Corr['Area'] = []
for i in range(len(All_Before_Corr)):
    Repeat_Corr['Before'].append(All_Before_Corr[i])
    Repeat_Corr['After'].append(All_After_Corr[i])
    Repeat_Corr['Area'].append('V1')
for i in range(len(All_Before_Corr_V2)):
    Repeat_Corr['Before'].append(All_Before_Corr_V2[i])
    Repeat_Corr['After'].append(All_After_Corr_V2[i])
    Repeat_Corr['Area'].append('V2')
Repeat_Data_Frame = pd.DataFrame(Repeat_Corr)
Selected_Repeat_DF = Repeat_Data_Frame.groupby('Area').sample(15000)
fig = sns.jointplot(data=Selected_Repeat_DF, x="Before", y="After", hue="Area",kind="kde",height = 8,fill =True,xlim = (-0.2,0.5),ylim = (-0.2,0.5),thres = 0.1,alpha = 0.6)
#fig = sns.jointplot(data=Selected_Repeat_DF, x="Before", y="After", hue="Area",height = 8,xlim = (-0.2,0.5),ylim = (-0.2,0.5),s=4)
fig.ax_joint.plot([-0.2,0.5],[-0.2,0.5], ls = '--',c = 'gray')
fig.savefig(work_path+r'\Kernel.png',dpi = 180)
#%% V2 Circle tunings. Circle-Triangle.
CP = Cell_Processor(r'K:\Test_Data\2P\210514_L76_2P')
t_data = CP.T_Map_Plot_Core('Run013', [17,18,19,20,21,22,23,24], [1,2,3,4,5,6,7,8])

