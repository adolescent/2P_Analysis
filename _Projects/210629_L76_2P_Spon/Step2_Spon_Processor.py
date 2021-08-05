# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 14:08:35 2021

@author: ZR
"""

import Spontaneous_Processor as SP


Sr = SP.Single_Run_Spontaneous_Processor(r'K:\Test_Data\2P\210629_L76_2P',
                                         spon_run = 'Run001')
PCA_Dic = Sr.Do_PCA(3700,9999)
Sr.Pairwise_Correlation_Plot(Sr.spon_cellname, 3700, 9999,'All_Before',cor_range = (-0.2,0.8))
Mu = SP.Multi_Run_Spontaneous_Processor(r'K:\Test_Data\2P\210629_L76_2P', 1.301)

#%% Evaluate cell fluctuation.
import Cell_2_DataFrame as C2D
import Cell_Train_Analyzer.Cell_Activity_Evaluator as CAE
All_Spon_Before = C2D.Multi_Run_Fvalue_Cat(r'K:\Test_Data\2P\210629_L76_2P', ['Run001','Run002','Run003'],rest_time = (600,600))
spike_count,Z_count = CAE.Spike_Count(All_Spon_Before)
# Get Z score tuning.
Spon_After = C2D.Single_Run_Fvalue_Frame(r'K:\Test_Data\2P\210629_L76_2P', 'Run005')
after_spike_count,after_Z = CAE.Spike_Count(Spon_After)
# Plot all tuning graphs.
save_folder = r'K:\Test_Data\2P\210629_L76_2P\_All_Results\Spon_Analyze'
import seaborn as sns
import matplotlib.pyplot as plt
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
import Analyzer.My_FFT as fft
import numpy as np
all_cell_before_avr = np.array(Z_count.mean(0))
all_cell_after_avr = np.array(after_Z.mean(0))
before_spectrum = fft.FFT_Window_Slide(all_cell_before_avr,window_length=120)
after_spectrum = fft.FFT_Window_Slide(all_cell_after_avr,window_length=120)
frequency_stick = np.array(before_spectrum.loc[:,'Frequency'])
before_time_stick = np.arange(0,318,1)

num_ticks = 14
yticks = np.linspace(0,len(frequency_stick)-1, num_ticks,dtype = np.int)
yticklabels = [frequency_stick[idx] for idx in yticks]
ax = plt.figure(figsize = (12,8))
ax = sns.heatmap(before_spectrum.iloc[:,1:],yticklabels = [0,'',1])
ax.set_yticks(yticks)
ax.set_yticklabels(np.round(yticklabels,2),rotation = 30)
ax.set_ylabel('Frequency(Hz)')


#%% Get Tuing property of this day's run.
from Stimulus_Cell_Processor.Tuning_Property_Calculator import Tuning_Property_Calculator
import OS_Tools_Kit as ot

Tuning_0629 = Tuning_Property_Calculator(r'K:\Test_Data\2P\210629_L76_2P',
                                         Orien_para = ('Run004','G8_2P'),
                                         OD_para = ('Run006','OD_2P'),
                                         Hue_para = ('Run007','RGLum',False))


ot.Save_Variable(r'K:\Test_Data\2P\210629_L76_2P', 'All_Cell_Tuning', Tuning_0629,'.tuning')
#%% Evaluate Tuing Property after stim.



