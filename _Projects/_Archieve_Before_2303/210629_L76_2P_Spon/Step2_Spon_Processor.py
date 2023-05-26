# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 14:08:35 2021

@author: ZR
"""
#Import 
import Spontaneous_Processor as SP
import Cell_2_DataFrame as C2D
import Series_Analyzer.Cell_Activity_Evaluator as CAE
from Series_Analyzer import Spontaneous_Preprocessing as Prepro
from Series_Analyzer import Pairwise_Corr
import OS_Tools_Kit as ot
from Plotter.Heatmap import Heat_Maps
from Plotter.Line_Plotter import EZLine
import seaborn as sns
import matplotlib.pyplot as plt
import Analyzer.My_FFT as fft
import numpy as np
from Stimulus_Cell_Processor.Tuning_Property_Calculator import Tuning_Property_Calculator
from Series_Analyzer import Cell_Frame_PCA as PCA
from Plotter.Hist_Plotter import Multi_Hist_Plot
import Plotter.Line_Plotter as Plo
from Statistic_Tools import T_Test_Pair
from Stimulus_Cell_Processor.Tuning_Selector import Get_Tuning_Checklists
import Cell_Processor as CP
import random

#%%

Sr = SP.Single_Run_Spontaneous_Processor(r'K:\Test_Data\2P\210629_L76_2P',
                                         spon_run = 'Run001')
PCA_Dic = Sr.Do_PCA(3700,9999)
Sr.Pairwise_Correlation_Plot(Sr.spon_cellname, 3700, 9999,'All_Before',cor_range = (-0.2,0.8))
Mu = SP.Multi_Run_Spontaneous_Processor(r'K:\Test_Data\2P\210629_L76_2P', 1.301)

#%% Evaluate cell fluctuation.

All_Spon_Before = C2D.Multi_Run_Fvalue_Cat(r'K:\Test_Data\2P\210629_L76_2P', ['Run001','Run002','Run003'],rest_time = (600,600))
spike_count,Z_count = CAE.Spike_Count(All_Spon_Before)
# Get Z score tuning.
Spon_After = C2D.Single_Run_Fvalue_Frame(r'K:\Test_Data\2P\210629_L76_2P', 'Run005')
after_spike_count,after_Z = CAE.Spike_Count(Spon_After)
# Plot all tuning graphs.
save_folder = r'K:\Test_Data\2P\210629_L76_2P\_All_Results\Spon_Analyze'

#Before Graph
fig = plt.figure(figsize = (25,15))
plt.title('dF Count Before Stim',fontsize=36)
fig = sns.heatmap(Z_count,square=False,yticklabels=False,center = 0)
fig.figure.savefig(save_folder+'\Count_Before.png')
plt.clf()
#After Graph
fig = plt.figure(figsize = (25,15))
plt.title('dF/F Count After Stim',fontsize=36)
fig = sns.heatmap(after_Z,square=False,yticklabels=False,center = 0)
fig.figure.savefig(save_folder+'\Count_After.png')
plt.clf()
#%% Calculate average freq spectrum
all_cell_before_avr = np.array(Z_count.mean(0))
all_cell_after_avr = np.array(after_Z.mean(0))
before_spectrum = fft.FFT_Window_Slide(all_cell_before_avr,window_length=120)
after_spectrum = fft.FFT_Window_Slide(all_cell_after_avr,window_length=120)
frequency_stick = np.array(before_spectrum.loc[:,'Frequency'])
before_time_stick = np.arange(0,318,1)
#%% Plot average graph of all cell before/after spontaneous.
all_before = Z_count.mean(0).to_frame()
before_time_series = np.arange(0,25018)/1.301/60
all_before.insert(0,'Times',before_time_series)
all_before.columns = ['Times (min)','Before dF/F Average']
ax = plt.figure(figsize = (20,10))
ax = sns.lineplot(data = all_before,x = 'Times (min)',y = 'Before dF/F Average')
plt.savefig('AVR_Before.png')

all_after = after_Z.mean(0).to_frame()
after_time_series = np.arange(0,4620)/1.301/60
all_after.insert(0,'Times',after_time_series)
all_after.columns = ['Times (min)','After dF/F Average']
ax = plt.figure(figsize = (20,10))
ax = sns.lineplot(data = all_after,x = 'Times (min)',y = 'After dF/F Average')
plt.savefig('AVR_After.png')

#%% Plot Before Stim Spon Spectrum
num_ticks = 14
yticks = np.linspace(0,len(frequency_stick)-1, num_ticks,dtype = np.int)
yticklabels = [frequency_stick[idx] for idx in yticks]
ax = plt.figure(figsize = (24,10))
plt.title('Power Spectrum Before Stim',fontsize = 25)
ax = sns.heatmap(before_spectrum.iloc[:,1:])
ax.set_yticks(yticks)
ax.set_yticklabels(np.round(yticklabels,2),rotation = 30)
ax.set_ylabel('Frequency(Hz)')
xticks = np.linspace(0,len(before_time_stick)-1, 18,dtype = np.int)
xticklabels = [before_time_stick[idx] for idx in xticks]
ax.set_xticks(xticks)
ax.set_xticklabels(np.round(xticklabels,2),rotation = 0)
ax.set_xlabel('Window Start Time(min)')
plt.savefig('Before_Spectrum.png')
#%% Plot After Stim Spon Spectrum
after_time_ticks = np.arange(0,60)
num_y_ticks = 14
num_x_ticks = 20
yticks = np.linspace(0,len(frequency_stick)-1, num_y_ticks,dtype = np.int)
yticklabels = [frequency_stick[idx] for idx in yticks]
ax = plt.figure(figsize = (24,10))
plt.title('Power Spectrum Aefore Stim',fontsize = 25)
ax = sns.heatmap(after_spectrum.iloc[:,1:])
ax.set_yticks(yticks)
ax.set_yticklabels(np.round(yticklabels,2),rotation = 30)
ax.set_ylabel('Frequency(Hz)')
xticks = np.linspace(0,len(after_time_ticks)-1,num_x_ticks,dtype = np.int)
xticklabels = [before_time_stick[idx] for idx in xticks]
ax.set_xticks(xticks)
ax.set_xticklabels(np.round(xticklabels,2),rotation = 0)
ax.set_xlabel('Window Start Time(min)')
plt.savefig('After_Spectrum.png')

#%% Get Tuing property of this day's run.


Tuning_0629 = Tuning_Property_Calculator(r'K:\Test_Data\2P\210629_L76_2P',
                                         Orien_para = ('Run004','G8_2P'),
                                         OD_para = ('Run006','OD_2P'),
                                         Hue_para = ('Run007','RGLum',False))


ot.Save_Variable(r'K:\Test_Data\2P\210629_L76_2P', 'All_Cell_Tuning', Tuning_0629,'.tuning')
#%% Do PCA for Spon Before..
day_folder = r'K:\Test_Data\2P\210629_L76_2P'
save_folder = r'K:\Test_Data\2P\210629_L76_2P\_All_Results\Spon_Analyze'

all_cell_dic = ot.Load_Variable(r'K:\Test_Data\2P\210629_L76_2P\L76_210629A_All_Cells.ac')
Before_Cell_Data = Prepro.Pre_Processor(day_folder,runname = 'Run002')
Before_PCs,Before_PC_info = PCA.Do_PCA(Before_Cell_Data)

Plo.EZLine(Before_PC_info['Accumulated_Variance_Ratio'],
           save_folder = save_folder,graph_name = 'ROC_Before',
           title = 'Accumulated Variance In Before PCs',
           figsize = (8,4),x_label = 'PCs',y_label = 'Ratio',
           y_range = (0,1.1))
ot.Save_Variable(save_folder, 'Before_PCA_Info', Before_PC_info)
PCA.Compoment_Visualize(Before_PCs, all_cell_dic, save_folder)

After_Cell_Data = Prepro.Pre_Processor(day_folder,runname = 'Run005')
After_PCs,After_PC_info = PCA.Do_PCA(After_Cell_Data)
Plo.EZLine(After_PC_info['Accumulated_Variance_Ratio'],
           save_folder = save_folder,graph_name = 'ROC_After',
           title = 'Accumulated Variance In After PCs',
           figsize = (8,4),x_label = 'PCs',y_label = 'Ratio',
           y_range = (0,1.1))
ot.Save_Variable(save_folder, 'After_PCA_Info', After_PC_info)
PCA.Compoment_Visualize(After_PCs, all_cell_dic, save_folder)
# Compare Before& After PCA ROC
Plo.Multi_Line_Plot([Before_PC_info['Accumulated_Variance'],After_PC_info['Accumulated_Variance']],
                    legends=['Variance Before','Variance After'],save_folder = save_folder,
                    graph_name = 'Variance Compare',title = 'PCA Explained Variance',
                    x_label='PCs',y_label = 'Explained Variance')
Plo.Multi_Line_Plot([Before_PC_info['Accumulated_Variance_Ratio'],After_PC_info['Accumulated_Variance_Ratio']],
                    legends=['Variance Before','Variance After'],save_folder = save_folder,
                    graph_name = 'Variance Compare2',title = 'PCA Explained Variance Ratio',
                    x_label='PCs',y_label = 'Explained Variance Ratio')

#%% Calculater pairwise correlation.

# to make Before & After comparable, we cut them to same series.
Whole_Series_corr_Before = Before_Cell_Data.iloc[:,0:4625]
Whole_Series_corr_After = After_Cell_Data
all_cell_names = Whole_Series_corr_Before.index.tolist()
Before_Pair_Corr = Pairwise_Corr.Pair_Corr_Core(Whole_Series_corr_Before,
                                                all_cell_names,set_name = 'All_Cell_Before')

After_Pair_Corr = Pairwise_Corr.Pair_Corr_Core(Whole_Series_corr_After,
                                                all_cell_names,set_name = 'All_Cell_After')

Multi_Hist_Plot([Before_Pair_Corr,After_Pair_Corr],label_lists = ['Before','After'],
                save_folder = save_folder,graph_name = 'Pair_Corr',
                title = 'Pairwise Correlation',x_label = 'spearman r')
t,p,D = T_Test_Pair(np.array(After_Pair_Corr).flatten(),np.array(Before_Pair_Corr).flatten())
#%% Pairwise corelation control
Before_A = Before_Cell_Data.iloc[:,0:4500]
Before_B = Before_Cell_Data.iloc[:,4500:9000]
Corr_A = Pairwise_Corr.Pair_Corr_Core(Before_A,
                                      all_cell_names,set_name = 'First 1h')
Corr_B = Pairwise_Corr.Pair_Corr_Core(Before_B,
                                      all_cell_names,set_name = 'Last 1h')

Multi_Hist_Plot([Corr_A,Corr_B],label_lists = ['Previous','Latter'],
                save_folder = save_folder,graph_name = 'Test',
                title = 'Pairwise Correlation Before Stimulus',x_label = 'spearman r')
#%% Calculate pairwise correlation of whole frame.
Before_Corr_Windowed = Pairwise_Corr.Pair_Corr_Window_Slide(Before_Cell_Data, all_cell_names)
ot.Save_Variable(save_folder, 'Pair_Corr_Train_Before', Before_Corr_Windowed)

After_Corr_Windowed = Pairwise_Corr.Pair_Corr_Window_Slide(After_Cell_Data, all_cell_names)
ot.Save_Variable(save_folder, 'Pair_Corr_Train_After', After_Corr_Windowed)
# Calculate pairwise correlation statistic and plot.
Before_Corr_Hist,Before_Corr_t = Pairwise_Corr.Corr_Histo(Before_Corr_Windowed)
After_Corr_Hist,After_Corr_t = Pairwise_Corr.Corr_Histo(After_Corr_Windowed)


Heat_Maps(Before_Corr_Hist,save_folder = save_folder,graph_name = 'Before_Corr',square = False,
          title = 'Pairwise Correlation Before Stimulus',x_label = 'Start Time(min)',
          y_label = 'Spearman r')
EZLine(Before_Corr_t,save_folder = save_folder,graph_name = 'Before_Corr_t',
       title = 'Before Spon t-test',x_label = 'Start Time(min)',y_label = 't value')

Heat_Maps(After_Corr_Hist,save_folder = save_folder,graph_name = 'After_Corr',square = False,
          title = 'Pairwise Correlation After Stimulus',x_label = 'Start Time(min)',
          y_label = 'Spearman r')
EZLine(After_Corr_t,save_folder = save_folder,graph_name = 'After_Corr_t',
       title = 'After Spon t-test',x_label = 'Start Time(min)',y_label = 't value')

#%% Generate all series correlation plot.

save_folder = r'K:\Test_Data\2P\210629_L76_2P\_All_Results\Spon_Analyze\Pairwise_Corr'
All_Series = C2D.Multi_Run_Fvalue_Cat(r'K:\Test_Data\2P\210629_L76_2P',
                                      runlists = ['Run001','Run002','Run003','Run004','Run005'],
                                      rest_time = (0,0,0,0))

Used_All_Series = Prepro.Pre_Processor_By_Frame(All_Series)
all_cell_name = Used_All_Series.index.tolist()
Whole_Corr_Windowed = Pairwise_Corr.Pair_Corr_Window_Slide(Used_All_Series, all_cell_name)
ot.Save_Variable(save_folder, 'Pair_Corr_Train_Whole', Whole_Corr_Windowed)
Whole_Corr_Hist,Whole_Corr_t = Pairwise_Corr.Corr_Histo(Whole_Corr_Windowed)
Heat_Maps(Whole_Corr_Hist,save_folder = save_folder,graph_name = 'Whole_Corr',square = False,
          title = 'Pairwise Correlation Before Stimulus',x_label = 'Start Time(min)',
          y_label = 'Spearman r')
EZLine(Whole_Corr_t,save_folder = save_folder,graph_name = 'Whole_Corr_t',
       title = 'Whole Series t-test',x_label = 'Start Time(min)',y_label = 't value')
#%% Partial cell pairwise correlations
Cp = CP.Cell_Processor(day_folder)
all_cell_name = Used_All_Series.index.tolist()
OD_map = Cp.T_Map_Plot_Core('Run006',[1,3,5,7],[2,4,6,8])
AO_map = Cp.T_Map_Plot_Core('Run004',[2,6],[4,8])
day_folder = r'K:\Test_Data\2P\210629_L76_2P'
tuning_checklists = Get_Tuning_Checklists(day_folder)
RE_Cells = tuning_checklists['RE']
Orien45_Cells = tuning_checklists['Orien45']
Orien135_Cells = tuning_checklists['Orien135']
rand_cell_for_RE = random.sample(all_cell_name, 128)
rand_cell_for_Orien45 = random.sample(all_cell_name, 127)
rand_cell_for_Orien135 = random.sample(all_cell_name, 113)
#%% Calculate RE cell pairwise correlation.
save_folder = r'K:\Test_Data\2P\210629_L76_2P\_All_Results\Spon_Analyze\Pairwise_Corr'
RE_Corr_Windowed = Pairwise_Corr.Pair_Corr_Window_Slide(Used_All_Series,RE_Cells)
ot.Save_Variable(save_folder, 'Pair_Corr_Train_RE',RE_Corr_Windowed)
RE_Corr_Hist,RE_Corr_t = Pairwise_Corr.Corr_Histo(RE_Corr_Windowed,corr_lim = (-0.6,0.66))
Heat_Maps(RE_Corr_Hist,save_folder = save_folder,graph_name = 'RE_Corr',square = False,
          title = 'Pairwise Correlation RE',x_label = 'Start Time(min)',
          y_label = 'Spearman r')
EZLine(RE_Corr_t,save_folder = save_folder,graph_name = 'RE_Corr_t',
       title = 'RE Series t-test',x_label = 'Start Time(min)',y_label = 't value')
RE_Rand_Corr_Windowed = Pairwise_Corr.Pair_Corr_Window_Slide(Used_All_Series,rand_cell_for_RE)
ot.Save_Variable(save_folder, 'Pair_Corr_Train_Random_RE',RE_Rand_Corr_Windowed)
RE_Rand_Corr_Hist,RE_Rand_t = Pairwise_Corr.Corr_Histo(RE_Rand_Corr_Windowed,corr_lim = (-0.6,0.66))
Heat_Maps(RE_Rand_Corr_Hist,save_folder = save_folder,graph_name = 'RE_Rand_Corr',square = False,
          title = 'Pairwise Correlation Random',x_label = 'Start Time(min)',
          y_label = 'Spearman r')
EZLine(RE_Rand_t,save_folder = save_folder,graph_name = 'RE_Rand_Corr_t',
       title = 'Random Series t-test',x_label = 'Start Time(min)',y_label = 't value')
RE_t,_ = Pairwise_Corr.Window_by_Window_T_Testor(RE_Corr_Windowed, RE_Rand_Corr_Windowed)
EZLine(RE_t,save_folder = save_folder,graph_name = 'RE_vs_Rand',
       title = 'RE vs Random t-test',x_label = 'Start Time(min)',y_label = 't value')


#%%Calculate Orien135,Orien45,Orien135rand,Orien45rand cells.
Orien45_Corr_Windowed = Pairwise_Corr.Pair_Corr_Window_Slide(Used_All_Series,Orien45_Cells)
ot.Save_Variable(save_folder, 'Whole_Pair_Corr_Train_Orien45',Orien45_Corr_Windowed)
Orien135_Corr_Windowed = Pairwise_Corr.Pair_Corr_Window_Slide(Used_All_Series,Orien135_Cells)
ot.Save_Variable(save_folder, 'Whole_Pair_Corr_Train_Orien135',Orien135_Corr_Windowed)
Orien45_Rand_Corr_Windowed = Pairwise_Corr.Pair_Corr_Window_Slide(Used_All_Series,rand_cell_for_Orien45)
ot.Save_Variable(save_folder, 'Whole_Pair_Corr_Train_Rand_Orien45',Orien45_Rand_Corr_Windowed)
Orien135_Rand_Corr_Windowed = Pairwise_Corr.Pair_Corr_Window_Slide(Used_All_Series,rand_cell_for_Orien135)
ot.Save_Variable(save_folder, 'Whole_Pair_Corr_Train_Rand_Orien135',Orien135_Rand_Corr_Windowed)
Orien45_Hist,Orien45_Corr_t =  Pairwise_Corr.Corr_Histo(Orien45_Corr_Windowed,corr_lim = (-0.6,0.66))
Orien45_Rand_Hist,Orien45_Rand_Corr_t = Pairwise_Corr.Corr_Histo(Orien45_Rand_Corr_Windowed,corr_lim = (-0.6,0.66))
Orien45_vs_rand_t,_ = Pairwise_Corr.Window_by_Window_T_Testor(Orien45_Corr_Windowed, Orien45_Rand_Corr_Windowed)
Orien135_vs_rand_t,_ = Pairwise_Corr.Window_by_Window_T_Testor(Orien135_Corr_Windowed, Orien135_Rand_Corr_Windowed)
EZLine(Orien45_vs_rand_t,save_folder = save_folder,graph_name = 'Orien45_vs_Rand',figsize = (15,8),
       title = 'Orien45 vs Random t-test',x_label = 'Start Time(min)',y_label = 't value')
EZLine(Orien135_vs_rand_t,save_folder = save_folder,graph_name = 'Orien135_vs_Rand',figsize = (15,8),
       title = 'Orien135 vs Random t-test',x_label = 'Start Time(min)',y_label = 't value')
Orien45_Corr_Hist,Orien45_t = Pairwise_Corr.Corr_Histo(Orien45_Corr_Windowed,corr_lim = (-0.6,0.66))
Heat_Maps(Orien45_Corr_Hist,save_folder = save_folder,graph_name = 'Orien45_Corr',square = False,
          title = 'Pairwise Correlation Orien45',x_label = 'Start Time(min)',
          y_label = 'Spearman r')
EZLine(Orien45_t,save_folder = save_folder,graph_name = 'Orien45_Corr_t',
       title = 'Orien45 Series t-test',x_label = 'Start Time(min)',y_label = 't value')
Orien135_Corr_Hist,Orien135_t = Pairwise_Corr.Corr_Histo(Orien135_Corr_Windowed,corr_lim = (-0.6,0.66))
Heat_Maps(Orien135_Corr_Hist,save_folder = save_folder,graph_name = 'Orien135_Corr',square = False,
          title = 'Pairwise Correlation Orien135',x_label = 'Start Time(min)',
          y_label = 'Spearman r')
EZLine(Orien135_t,save_folder = save_folder,graph_name = 'Orien135_Corr_t',
       title = 'Orien135 Series t-test',x_label = 'Start Time(min)',y_label = 't value')
#%% Calculate selected Orien135-45 cells, selected Orien45-RE cells
orien_num = 113
od_num = 127
selected_orien45_cells = random.sample(Orien45_Cells,orien_num)
selected_re_cells = random.sample(RE_Cells, od_num)
Orien45_For_Compare = Pairwise_Corr.Pair_Corr_Window_Slide(Used_All_Series,selected_orien45_cells)
ot.Save_Variable(save_folder, 'Whole_Pair_Corr_Train_Orien45_113Cells',Orien45_For_Compare)
RE_For_Compare = Pairwise_Corr.Pair_Corr_Window_Slide(Used_All_Series,selected_re_cells)
ot.Save_Variable(save_folder, 'Whole_Pair_Corr_Train_RE_127Cells',RE_For_Compare)
RE_vs_Orien45,_ = Pairwise_Corr.Window_by_Window_T_Testor(RE_Corr_Windowed,Orien45_For_Compare)
Orien135_vs_Orien45,_ = Pairwise_Corr.Window_by_Window_T_Testor(Orien135_Corr_Windowed,Orien45_For_Compare)
EZLine(RE_vs_Orien45,save_folder = save_folder,graph_name = 'RE_vs_Orien45_t',
       title = 'RE vs Orien45 t-test',x_label = 'Start Time(min)',y_label = 't value')
EZLine(Orien135_vs_Orien45,save_folder = save_folder,graph_name = 'Orien135_vs_Orien45_t',
       title = 'Orien135 vs Orien45 t-test',x_label = 'Start Time(min)',y_label = 't value')

