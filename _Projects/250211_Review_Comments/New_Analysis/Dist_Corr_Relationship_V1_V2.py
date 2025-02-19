'''
Compare distance-correlation relationship between all locations in v1 and v2.
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
from Classifier_Analyzer import *
from Cell_Class.Timecourse_Analyzer import *
from Review_Fix_Funcs import *
from Filters import Signal_Filter_v2
import warnings

warnings.filterwarnings("ignore")

savefolder = r'D:\_GoogleDrive_Files\#Figs\#250211_Revision1\Dist_Corr_Relation'
paircorr_v2 = ot.Load_Variable_v2(r'D:\_GoogleDrive_Files\#Figs\#250211_Revision1\Fig5','All_Pair_Corrs_V2.pkl')
paircorr_v1 = ot.Load_Variable_v2(r'D:\_GoogleDrive_Files\#Figs\#250211_Revision1\Fig4','All_Pair_Corrs.pkl')

#%%
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

v1_locs = list(paircorr_v1.keys())
v2_locs = list(paircorr_v2.keys())
exp_fit_paras = pd.DataFrame(columns = ['Loc','Area','A','B','C','R2'])

def exponential_model(dist,A,  B, C):
    return A*np.exp(-B * dist) + C


for i,cloc in enumerate(v1_locs):
    c_corr = paircorr_v1[cloc]
    dists = c_corr['Dist']
    corr = c_corr['Corr']
    params, covariance = curve_fit(exponential_model,dists,corr,p0=[1,0.1,0.3])
    A, B, C = params
    predicted_corr = exponential_model(dists, A, B, C)
    r_squared = r2_score(corr, predicted_corr)
    exp_fit_paras.loc[len(exp_fit_paras)] = [cloc,'V1',A,B,C,r_squared]

for i,cloc in enumerate(v2_locs):
    c_corr = paircorr_v2[cloc]
    dists = c_corr['Dist']
    corr = c_corr['Corr']
    params, covariance = curve_fit(exponential_model,dists,corr,p0=[1,0.1,0.3])
    A, B, C = params
    predicted_corr = exponential_model(dists, A, B, C)
    r_squared = r2_score(corr, predicted_corr)
    exp_fit_paras.loc[len(exp_fit_paras)] = [cloc,'V2',A,B,C,r_squared]

exp_fit_paras['Decay_Dist'] = 1/exp_fit_paras['B']
# c_corr = paircorr_v1[v1_locs[2]]
# dists = c_corr['Dist']
# corr = c_corr['Corr']
# sns.scatterplot(x = dists,y=corr,s = 3,lw=0)

#%% fit dist and corr example
c_corr = paircorr_v2[v2_locs[2]]
# c_corr = paircorr_v1[v1_locs[4]]

dists = c_corr['Dist']
corr = c_corr['Corr']

def exponential_model(dist, A, B, C):
    return A * np.exp(-B * dist) + C

params, covariance = curve_fit(
    exponential_model, 
    dists, 
    corr, 
    p0=[1,0.1,0.3]
)
A, B, C = params

sns.scatterplot(x = dists,y=corr,s = 3,lw=0)
x = np.arange(600)
y = exponential_model(x,*params)
plt.plot(y,c='r')

predicted_corr = exponential_model(dists, A, B, C)
r_squared = r2_score(corr, predicted_corr)
print(f'r2={r_squared}')

#%% plot decay distance
fig,ax = plt.subplots(ncols=1,nrows=1,figsize = (3,5))
sns.barplot(data=exp_fit_paras,y = 'Decay_Dist',x='Area',hue='Area',width=0.5,capsize=0.1)

