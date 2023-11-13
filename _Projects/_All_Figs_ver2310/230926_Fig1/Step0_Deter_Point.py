'''
This script will determine the start point and end point of spon data capture.

We use 100 binned data std, when they first above 1 as the time of recording.

'''
#%%

from Cell_Class.Stim_Calculators import Stim_Cells
from Cell_Class.Format_Cell import Cell
import OS_Tools_Kit as ot
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from Plotter.Line_Plotter import EZLine
from tqdm import tqdm

#%% V1 use thres 1 and window 100.
V1_folder = r'D:\_All_Spon_Datas_V1'
run_folders = ot.Get_Sub_Folders(V1_folder)
work_path = r'D:\_Path_For_Figs\Fig0_Timepoints'

window_size = 100
thres_std = 1
for i,c_run in tqdm(enumerate(run_folders)):
    c_name = c_run.split('\\')[-1]
    ac = ot.Load_Variable_v2(c_run,'Cell_Class.pkl')
    c_spon = ac.Z_Frames['1-001']
    # group dff with length 100
    grouped_spon = c_spon.groupby({x: x // window_size for x in range(len(c_spon))})
    std_series = grouped_spon.std().iloc[:-4,:] # N_window*M_Cell,and ignore last 200 frame.
    start_timepoints = np.where(std_series.mean(1)>thres_std)[0][0]
    c_usedspon = c_spon.iloc[start_timepoints*window_size:-400,:]
    ot.Save_Variable(c_run,'Spon_Before',c_usedspon)
    
    # pivot frame.
    series_plotable = pd.melt(std_series.T,value_name = 'Cell Std',var_name = 'Time window')
    fig,ax = plt.subplots(figsize=(8,4),dpi = 180)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    ax.axvspan(xmin = start_timepoints,xmax = series_plotable.max()[0],alpha = 0.2,color='r') # fill used part of data in red.
    ax = sns.lineplot(data = series_plotable,x = 'Time window',y = 'Cell Std')
    # ax.axvline(x = start_timepoints,c = 'r')# add a determine point of start
    ax.set_title('Std Curve of '+ c_name)
    fig.savefig(c_run+'\\'+c_name+'_Used_Spon.png',bbox_inches='tight')
    fig.savefig(work_path+'\\'+c_name+'_Used_Spon.png',bbox_inches='tight') # a stats backup.

#%% V2 use the same process.
V2_folder = r'D:\_All_Spon_Datas_V2'
run_folders = ot.Get_Sub_Folders(V2_folder)
work_path = r'D:\_Path_For_Figs\Fig0_Timepoints'

window_size = 100
thres_std = 1
for i,c_run in tqdm(enumerate(run_folders)):
    c_name = c_run.split('\\')[-1]
    ac = ot.Load_Variable_v2(c_run,'Cell_Class.pkl')
    c_spon = ac.Z_Frames['1-001']
    # group dff with length 100
    grouped_spon = c_spon.groupby({x: x // window_size for x in range(len(c_spon))})
    std_series = grouped_spon.std().iloc[:-4,:] # N_window*M_Cell,and ignore last 200 frame.
    start_timepoints = np.where(std_series.mean(1)>thres_std)[0][0]
    c_usedspon = c_spon.iloc[start_timepoints*window_size:-400,:]
    ot.Save_Variable(c_run,'Spon_Before',c_usedspon)
    
    # pivot frame.
    series_plotable = pd.melt(std_series.T,value_name = 'Cell Std',var_name = 'Time window')
    fig,ax = plt.subplots(figsize=(8,4),dpi = 180)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    ax.axvspan(xmin = start_timepoints,xmax = series_plotable.max()[0],alpha = 0.2,color='r') # fill used part of data in red.
    ax = sns.lineplot(data = series_plotable,x = 'Time window',y = 'Cell Std')
    # ax.axvline(x = start_timepoints,c = 'r')# add a determine point of start
    ax.set_title('Std Curve of '+ c_name)
    fig.savefig(c_run+'\\'+c_name+'_Used_Spon.png',bbox_inches='tight')
    fig.savefig(work_path+'\\'+c_name+'_Used_Spon.png',bbox_inches='tight') # a stats backup.
