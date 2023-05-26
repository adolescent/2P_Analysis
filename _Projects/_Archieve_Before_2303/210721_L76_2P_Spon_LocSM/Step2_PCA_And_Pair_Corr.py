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

#%% Spon strengh evaluator
from Cell_2_DataFrame import Single_Run_Fvalue_Frame,Multi_Run_Fvalue_Cat
from Series_Analyzer.Cell_Activity_Evaluator import Spike_Count
from Plotter import Line_Plotter,Heatmap,Hist_Plotter
import numpy as np

day_folder = r'K:\Test_Data\2P\210721_L76_2P'
save_folder = r'K:\Test_Data\2P\210721_L76_2P\_All_Results'
Spon_Before_raw = Single_Run_Fvalue_Frame(day_folder,'Run001')
Before_spikes,Before_Z = Spike_Count(Spon_Before_raw)
Spon_After_raw = Single_Run_Fvalue_Frame(day_folder, 'Run003')
After_spikes,After_Z = Spike_Count(Spon_After_raw)
all_before_avr = Before_Z.mean(0).tolist()
timelist_before = np.arange(14065)/1.301/60
all_after_avr = After_Z.mean(0).tolist()
timelist_after = np.arange(9362)/1.301/60
Line_Plotter.EZLine(all_before_avr,x_series = timelist_before,save_folder = save_folder,graph_name='dF_Count_All_Cell_Before',
                    title='Before dF count',x_label='Time(min)',y_label = 'dF/F Count')
Line_Plotter.EZLine(all_after_avr,x_series = timelist_after,save_folder = save_folder,graph_name='dF_Count_All_Cell_After',
                    title='After dF count',x_label='Time(min)',y_label = 'dF/F Count')
# Plot heatmap
Heatmap.Heat_Maps(Before_Z,yticklabels = False,center=0,save_folder=save_folder,graph_name='Heatmap_Before',
                  title = 'dF/F Count Before',x_label = 'Frame',y_label = 'Cells',x_tick_num = 20)
Heatmap.Heat_Maps(After_Z,yticklabels = False,center=0,save_folder=save_folder,graph_name='Heatmap_After',
                  title = 'dF/F Count After',x_label = 'Frame',y_label = 'Cells',x_tick_num = 20)
All_Runs_raw = Multi_Run_Fvalue_Cat(day_folder, ['Run001','Run002','Run003'],rest_time=(0,0))
All_spikes,All_Z = Spike_Count(All_Runs_raw,window = 300,win_step = 60)
all_avr = All_Z.mean(0).tolist()
Line_Plotter.EZLine(all_avr,save_folder = save_folder,graph_name='All_avr_dF_F',title = 'Count All',
                    x_label = 'Time(min)',y_label = 'dF/F_Count')
#%% FFT Spectrum
import Analyzer.My_FFT as fft
before_spectrum = fft.FFT_Window_Slide(all_before_avr,window_length=120)
after_spectrum = fft.FFT_Window_Slide(all_after_avr,window_length=120)
Heatmap.Heat_Maps(before_spectrum,save_folder=save_folder,graph_name = 'Power_Spectrum_Before',
                  title = 'Power Spectrum Before',x_label = 'Time(min)',y_label = 'Frequency(Hz)',figsize= (25,15))
Heatmap.Heat_Maps(after_spectrum,save_folder=save_folder,graph_name = 'Power_Spectrum_After',
                  title = 'Power Spectrum After',x_label = 'Time(min)',y_label = 'Frequency(Hz)',figsize= (25,15))
#%% PCA in before & after
from Series_Analyzer.Spontaneous_Preprocessing import Pre_Processor as prepro
from Series_Analyzer.Cell_Frame_PCA import Do_PCA,Compoment_Visualize
import OS_Tools_Kit as ot
PCA_folder = save_folder+r'\PCA'
all_cell_dic = ot.Load_Variable(day_folder,'L76_210721A_All_Cell_Include_Run03.ac')
Before_dF_F_Train = prepro(day_folder,runname = 'Run001')
After_dF_F_Train = prepro(day_folder,runname = 'Run003')
before_PC_Comp,before_PC_info,_ = Do_PCA(Before_dF_F_Train)
_ = Compoment_Visualize(before_PC_Comp, all_cell_dic, PCA_folder)
ot.Save_Variable(PCA_folder, 'Before_Info', before_PC_info)
after_PC_Comp,after_PC_info,_ = Do_PCA(After_dF_F_Train)
_ = Compoment_Visualize(after_PC_Comp, all_cell_dic, PCA_folder)
ot.Save_Variable(PCA_folder, 'After_Info', after_PC_info)
Line_Plotter.Multi_Line_Plot([np.array(before_PC_info['Accumulated_Variance'][:228])/280,np.array(after_PC_info['Accumulated_Variance'])/227], 
                             legends = ['Before','After'],save_folder = PCA_folder,graph_name='Accu_Var_Before_After',
                             title = 'Accumulated Variance',x_label = 'PCs',y_label = 'Accumulated Variance')
Line_Plotter.Multi_Line_Plot([before_PC_info['Accumulated_Variance_Ratio'][:228],after_PC_info['Accumulated_Variance_Ratio']], 
                             legends = ['Before','After'],save_folder = PCA_folder,graph_name='Accu_Var_ratio_Before_After',
                             title = 'Accumulated Variance Ratio',x_label = 'PCs',y_label = 'Accumulated Variance Ratio')
#%% Pairwise correlation
from Stimulus_Cell_Processor.Tuning_Selector import Get_Tuning_Checklists
from Series_Analyzer.Pairwise_Corr import Pair_Corr_Window_Slide,Corr_Histo,Window_by_Window_T_Testor
from Plotter.Heatmap import Heat_Maps
from Plotter.Line_Plotter import EZLine
from Cell_2_DataFrame import Multi_Run_Fvalue_Cat
from Series_Analyzer.Spontaneous_Preprocessing import Pre_Processor_By_Frame
import random

tuning_checklist = Get_Tuning_Checklists(day_folder)
before_cell_names = Before_dF_F_Train.index.tolist()
all_corr_before = Pair_Corr_Window_Slide(Before_dF_F_Train, before_cell_names)
all_before_hist,all_before_t = Corr_Histo(all_corr_before)
Heat_Maps(all_before_hist,save_folder=save_folder,title = 'Pairwise Correlation Before',
          graph_name='Run01_Pair_Corr_All_Cell',x_label='Time(min)',y_label = 'Spearman r')
EZLine(all_before_t,save_folder = save_folder,graph_name = 'Run01_Pair_Corr_All_Cell_t',
       title = 'Before Spon t-test',x_label = 'Time(min)',y_label='t value')
after_cell_names = After_dF_F_Train.index.tolist()
all_corr_after = Pair_Corr_Window_Slide(After_dF_F_Train, after_cell_names)
ot.Save_Variable(save_folder,'Run01_All_Cell_Paircc', all_corr_before)
ot.Save_Variable(save_folder,'Run03_All_Cell_Paircc', all_corr_after)
whole_series = Multi_Run_Fvalue_Cat(day_folder, ['Run001','Run002','Run003'],rest_time = (0,0))
all_series_dF_F = Pre_Processor_By_Frame(whole_series)
all_cells = all_series_dF_F.index.tolist()
LE_cells = list(set(tuning_checklist['LE'])&set(all_cells))
RE_cells = list(set(tuning_checklist['RE'])&set(all_cells))
orien45_cells = list(set(tuning_checklist['Orien45'])&set(all_cells))
orien90_cells = list(set(tuning_checklist['Orien90'])&set(all_cells))
rd_for_LE = random.sample(all_cells, 131)
rd_for_RE = random.sample(all_cells, 63)
rd_for_orien45 = random.sample(all_cells, 162)
rd_for_orien90 = random.sample(all_cells, 125)
LE_for_compare = random.sample(LE_cells, 125)

#%% Get all useful pairwise corrs.
all_corr_whole = Pair_Corr_Window_Slide(all_series_dF_F, all_cells)
ot.Save_Variable(save_folder, 'Run01_03_all_corr', all_corr_whole)

LE_corr_whole = Pair_Corr_Window_Slide(all_series_dF_F, LE_cells)
ot.Save_Variable(save_folder, 'Run01_03_LE_corr',LE_corr_whole)

RE_corr_whole = Pair_Corr_Window_Slide(all_series_dF_F, RE_cells)
ot.Save_Variable(save_folder, 'Run01_03_RE_corr', RE_corr_whole)

Orien45_corr_whole = Pair_Corr_Window_Slide(all_series_dF_F, orien45_cells)
ot.Save_Variable(save_folder, 'Run01_03_Orien45_corr', Orien45_corr_whole)

Orien90_corr_whole = Pair_Corr_Window_Slide(all_series_dF_F, orien90_cells)
ot.Save_Variable(save_folder, 'Run01_03_Orien90_corr', Orien90_corr_whole)

LE_for_compare = Pair_Corr_Window_Slide(all_series_dF_F, LE_for_compare)
ot.Save_Variable(save_folder, 'Run01_03_LE125_corr', LE_for_compare)

rd_LE_corr_whole = Pair_Corr_Window_Slide(all_series_dF_F, rd_for_LE)
ot.Save_Variable(save_folder, 'Run01_03_LErd_corr', rd_LE_corr_whole)

rd_RE_corr_whole = Pair_Corr_Window_Slide(all_series_dF_F, rd_for_RE)
ot.Save_Variable(save_folder, 'Run01_03_RErd_corr', rd_RE_corr_whole)

rd_orien45_corr_whole = Pair_Corr_Window_Slide(all_series_dF_F, rd_for_orien45)
ot.Save_Variable(save_folder, 'Run01_03_Orien45rd_corr', rd_orien45_corr_whole)

rd_orien90_whole = Pair_Corr_Window_Slide(all_series_dF_F,rd_for_orien90)
ot.Save_Variable(save_folder, 'Run01_03_Orien90rd_corr', rd_orien90_whole)

LE_corr_after = Pair_Corr_Window_Slide(After_dF_F_Train, LE_cells)
ot.Save_Variable(save_folder, 'Run03_LE_corr', LE_corr_after)

rd_LE_corr_after = Pair_Corr_Window_Slide(After_dF_F_Train, rd_for_LE)
ot.Save_Variable(save_folder, 'Run03_LErd_corr', rd_LE_corr_after)

Orien90_corr_after = Pair_Corr_Window_Slide(After_dF_F_Train, orien90_cells)
ot.Save_Variable(save_folder, 'Run03_Orien90_corr',Orien90_corr_after )

rd_Orien90_corr_after = Pair_Corr_Window_Slide(After_dF_F_Train, rd_for_orien90)
ot.Save_Variable(save_folder, 'Run03_Orien90_corr',rd_Orien90_corr_after )
#%% After get all pair corrs, plot them and calculate t value.
After_all_cell_hist,after_all_cell_t = Corr_Histo(all_corr_after,corr_lim=(-0.3,0.4))
Heat_Maps(After_all_cell_hist,save_folder = save_folder,graph_name = 'Run03_Pair_Corr_All_Cell',
          title = 'Pairwise Correlation After',x_label = 'Time(min)',y_label = 'Spearman r')
EZLine(after_all_cell_t,save_folder = save_folder,graph_name = 'Run03_Pair_Corr_All_Cell_t',
       title = 'After Spon t test',x_label = 'Time(min)',y_label='t value')

Whole_all_cell_hist,whole_all_cell_t = Corr_Histo(all_corr_whole,corr_lim = (-0.3,0.7))
Heat_Maps(Whole_all_cell_hist,save_folder = save_folder,graph_name = 'Run01-03_Pair_Corr_All_Cell',
          title = 'Pairwise Correlation Run01-03',x_label = 'Time(min)',y_label = 'Spearman r')
EZLine(whole_all_cell_t,save_folder = save_folder,graph_name = 'Run01-03_Pair_Corr_All_Cell_t',
       title = 'Run01-03 Spon t test',x_label = 'Time(min)',y_label='t value')
from Cell_Processor import Cell_Processor
Cp = Cell_Processor(day_folder)	
_ = Cp.T_Map_Plot_Core('Run006',[1,3,5,7],[2,4,6,8])
_ = Cp.T_Map_Plot_Core('Run002', [1,9], [5,13])
_ = Cp.T_Map_Plot_Core('Run002', [3,11], [7,15])
Whole_LE_hist,Whole_LE_t = Corr_Histo(LE_corr_whole,corr_lim = (-0.3,0.7))
Heat_Maps(Whole_LE_hist,save_folder = save_folder,graph_name = 'Run01-03_Pair_Corr_LE_Cell',
          title = 'Pairwise Correlation Run01-03 LE',x_label = 'Time(min)',y_label = 'Spearman r')
EZLine(Whole_LE_t,save_folder = save_folder,graph_name = 'Run01-03_Pair_Corr_LE_Cell_t',
       title = 'Run01-03 t test',x_label = 'Time(min)',y_label='t value')
Whole_LErd_hist,Whole_LErd_t = Corr_Histo(rd_LE_corr_whole,corr_lim = (-0.3,0.7))
Heat_Maps(Whole_LErd_hist,save_folder = save_folder,graph_name = 'Run01-03_Pair_Corr_LErd_Cell',
          title = 'Pairwise Correlation Run01-03 LErd',x_label = 'Time(min)',y_label = 'Spearman r')
EZLine(Whole_LErd_t,save_folder = save_folder,graph_name = 'Run01-03_Pair_Corr_LErd_Cell_t',
       title = 'Run01-03 t test',x_label = 'Time(min)',y_label='t value')
LE_vs_rd,_ = Window_by_Window_T_Testor(LE_corr_whole, rd_LE_corr_whole)
EZLine(LE_vs_rd,save_folder = save_folder,graph_name = 'Run01-03_LE_vs_rd_Window_t',
       title = 'Run01-03 LE_vs_RD',x_label = 'Time(min)',y_label='t value')

Whole_orien45_hist,Whole_orien45_t = Corr_Histo(Orien45_corr_whole,corr_lim = (-0.3,0.7))
Heat_Maps(Whole_orien45_hist,save_folder = save_folder,graph_name = 'Run01-03_Pair_Corr_Orien45_Cell',
          title = 'Pairwise Correlation Run01-03 Orien45',x_label = 'Time(min)',y_label = 'Spearman r')
EZLine(Whole_orien45_t,save_folder = save_folder,graph_name = 'Run01-03_Pair_Corr_Orien45_Cell_t',
       title = 'Run01-03 t test',x_label = 'Time(min)',y_label='t value')
Orien45_vs_rd,_ = Window_by_Window_T_Testor(Orien45_corr_whole, rd_orien45_corr_whole,thres = 1)
EZLine(Orien45_vs_rd,save_folder = save_folder,graph_name = 'Run01-03_Orien45_vs_rd_Window_t',
       title = 'Run01-03 Orien45_vs_RD',x_label = 'Time(min)',y_label='t value')

Whole_orien90_hist,Whole_orien90_t = Corr_Histo(Orien90_corr_whole,corr_lim = (-0.3,0.7))
Heat_Maps(Whole_orien90_hist,save_folder = save_folder,graph_name = 'Run01-03_Pair_Corr_Orien90_Cell',
          title = 'Pairwise Correlation Run01-03 Orien90',x_label = 'Time(min)',y_label = 'Spearman r')
EZLine(Whole_orien90_t,save_folder = save_folder,graph_name = 'Run01-03_Pair_Corr_Orien90_Cell_t',
       title = 'Run01-03 t test',x_label = 'Time(min)',y_label='t value')
Orien90_vs_rd,_ = Window_by_Window_T_Testor(Orien90_corr_whole, rd_orien90_whole,thres = 1)
EZLine(Orien90_vs_rd,save_folder = save_folder,graph_name = 'Run01-03_Orien90_vs_rd_Window_t',
       title = 'Run01-03 Orien90_vs_RD',x_label = 'Time(min)',y_label='t value')

LE_vs_orien90,_ = Window_by_Window_T_Testor(LE_for_compare, Orien90_corr_whole,thres = 1)
EZLine(LE_vs_orien90,save_folder = save_folder,graph_name = 'Run01-03_LE_vs_Orien90_t',
       title = 'Run01-03 LE_vs_Orien90',x_label = 'Time(min)',y_label='t value',y_range = (-5,25))


