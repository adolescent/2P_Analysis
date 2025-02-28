'''
Plot correlation decay with distance increase. Only 1D plot.

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
import cv2
from sklearn.model_selection import cross_val_score
from sklearn import svm
from scipy.stats import pearsonr
import scipy.stats as stats
from Cell_Class.Plot_Tools import Plot_3D_With_Labels
import copy
from Cell_Class.Advanced_Tools import *
from Cell_Class.Classifier_Analyzer import *
from Cell_Class.Timecourse_Analyzer import *
from Review_Fix_Funcs import *
from Filters import Signal_Filter_v2
import warnings

warnings.filterwarnings("ignore")

wp = r'D:\_GoogleDrive_Files\#Figs\#250211_Revision1\Fig4'
all_pair_corrs = ot.Load_Variable(wp,'All_Pair_Corrs.pkl')
all_locs = list(all_pair_corrs.keys())

cloc = all_locs[2]
example_corr = all_pair_corrs[cloc]


#%%
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import matplotlib.lines as mlines
cloc = all_locs[1]
example_corr = all_pair_corrs[cloc]

# fit exponential model for distance corr decay.
def exponential_model(dist,A,  B, C):
    return A*np.exp(-B * dist) + C
params, covariance = curve_fit(
    exponential_model, 
    example_corr['Dist'], 
    example_corr['Corr'], 
    p0=[1,0.1,0.3]
)
A, B, C = params
predicted_corr = exponential_model(example_corr['Dist'], A, B, C)
r_squared = r2_score(example_corr['Corr'], predicted_corr)
print(f'r2={r_squared}')


# plot scatter and fitted graph
plt.style.use('dark_background')
fig,ax = plt.subplots(nrows=1,ncols=1,figsize = (4,3),dpi=300)
# sns.scatterplot(data=example_corr,x='Dist',y='Corr',s=2,alpha=0.8,ax=ax,lw=0)
x = np.arange(600)
y = exponential_model(x,*params)
ax.plot(x,y,c='yellow')
pix_dist = 830/512
decay_pix = np.log(2)/params[1]
est_height = exponential_model(decay_pix,*params)
# ax.axvline(x=decay_pix,ymin=0,ymax=est_height,linestyle='--',color='gray')
line = mlines.Line2D([decay_pix,decay_pix], [0,est_height], label='5%', color='r', linestyle='--', linewidth=1)
ax.add_line(line)
# ax.set_ylabel('')
# ax.set_xlabel('')

ax.set_ylim(0.15,0.4)
ax.set_xlim(0,600)
# # ax.set_ylim(0,1)
ax.set_xticks([0/pix_dist,300/pix_dist,600/pix_dist,900/pix_dist])

ax.set_xticklabels([0,300,600,900],fontsize = 18)
ax.set_yticks([0.2,0.4])
ax.set_yticklabels([0.2,0.4],fontsize = 18)

fig.savefig(r'D:\_GoogleDrive_Files\#Figs\#250211_Revision1\Fig4\Insert_Dist.png',bbox_inches = 'tight') 
#%%
all_loc_infos = pd.DataFrame(columns = ['Loc','B','Dist'])
for i,cloc in enumerate(all_locs):

    example_corr = all_pair_corrs[cloc]
    params, covariance = curve_fit(exponential_model,example_corr['Dist'],example_corr['Corr'],p0=[1,0.1,0.3])
    B = params[1]
    dist = pix_dist*np.log(2)/B
    all_loc_infos.loc[len(all_loc_infos)] = [cloc,B,dist]

