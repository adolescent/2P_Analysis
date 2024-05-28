'''
This script will show all Z values of all locations cell.
We will also generate a matrix to keep this info.

'''

#%%

from Cell_Class.Stim_Calculators import Stim_Cells
from Cell_Class.Format_Cell import Cell
import OS_Tools_Kit as ot
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr
import scipy.stats as stats
import copy
from scipy.signal import find_peaks,peak_widths

savepath = r'D:\_Path_For_Figs\240520_Figs_ver_F1\Fig1_Brief'
datapath = r'D:\_All_Spon_Data_V1'
all_path_dic = list(ot.Get_Subfolders(r'D:\_All_Spon_Data_V1'))
all_path_dic.pop(4)
all_path_dic.pop(6)

#%%
'''
First, we will generate all Z value for all spon and all orien Z scores.
'''

for i,cloc in tqdm(enumerate(all_path_dic)):
    # ac = ot.Load_Variable(cloc,'Cell_Class.pkl')
    cloc_name = cloc.split('\\')[-1]
    c_spon = np.array(ot.Load_Variable(cloc,'Spon_Before.pkl'))
    # c_orien = np.array(ac.Z_Frames[ac.orienrun])
    c_z = np.array(c_spon).flatten()
    c_z_matrix = pd.DataFrame(c_z,columns = ['Z value'])
    c_z_matrix['Run'] = 'Spon'
    c_z_matrix['Loc'] = cloc_name
    if i == 0:
        all_z_matrix = copy.deepcopy(c_z_matrix)
    else:
        all_z_matrix = pd.concat((all_z_matrix,c_z_matrix))

for i,cloc in tqdm(enumerate(all_path_dic)):
    ac = ot.Load_Variable(cloc,'Cell_Class.pkl')
    cloc_name = cloc.split('\\')[-1]
    # c_spon = np.array(ot.Load_Variable(cloc,'Spon_Before.pkl'))
    c_orien = np.array(ac.Z_Frames[ac.orienrun])
    c_z = np.array(c_orien).flatten()
    c_z_matrix = pd.DataFrame(c_z,columns = ['Z value'])
    c_z_matrix['Run'] = 'Stimulus'
    c_z_matrix['Loc'] = cloc_name
    all_z_matrix = pd.concat((all_z_matrix,c_z_matrix))

ot.Save_Variable(savepath,'All_Z_Value',all_z_matrix)
#%%
'''
Second, we try to plot it. we will get the median and distribution of both stim and spon.
'''
plotable = all_z_matrix[all_z_matrix['Run']=='Stimulus']


median = plotable['Z value'].median()
plt.clf()
plt.cla()
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (5,5),dpi = 180)
sns.histplot(plotable,x = 'Z value',bins = np.linspace(-3,5,25),ax = ax)
ax.axvline(x = median,color = 'gray',linestyle = '--')

ax.set_title('Stimulus Z Scores')
ax.set_ylabel('Count')
ax.set_xlabel('Z Score')
