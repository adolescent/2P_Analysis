'''
This is all time-course distribution of 
'''

#%%
import OS_Tools_Kit as ot
import numpy as np
import pandas as pd
from Standard_Parameters.Sub_Graph_Dics import Sub_Dic_Generator
import Graph_Operation_Kit as gt
import cv2
import umap
import umap.plot
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from Kill_Cache import kill_all_cache
from sklearn.model_selection import cross_val_score
from sklearn import svm
from Cell_Tools.Cell_Visualization import Cell_Weight_Visualization
import random
import seaborn as sns
from My_Wheels.Cell_Class.Stim_Calculators import Stim_Cells
from My_Wheels.Cell_Class.Format_Cell import Cell
from Cell_Class.Advanced_Tools import *
from Cell_Class.Plot_Tools import *
from scipy.stats import pearsonr
import scipy.stats as stats

wp = r'D:\ZR\_Data_Temp\_All_Cell_Classes\220630_L76'
ac = ot.Load_Variable(wp,'Cell_Class.pkl')
spon_frame = ac.Z_Frames['1-001']
kill_all_cache(r'C:\ProgramData\anaconda3\envs\umapzr')
all_stim_frame,all_stim_label = ac.Combine_Frame_Labels(isi = True)
#%% Get unsupervised cluster, and count number of repeatance.
reducer = umap.UMAP(n_components=3,n_neighbors=20,target_weight=0)
reducer.fit(all_stim_frame,all_stim_label)
# ot.Save_Variable(wp,'Stim_All_UMAP_Unsup_3d',reducer)
u = reducer.embedding_
plt.switch_backend('webAgg')
fig,ax = Plot_3D_With_Labels(u,all_stim_label)
plt.show()
#%% get spon labels
spon_embedding = reducer.transform(spon_frame)
classifier,score = SVM_Classifier(embeddings = u,label = all_stim_label)
spon_label_unsup = SVC_Fit(classifier,spon_embedding,0)
plt.switch_backend('webAgg')
fig,ax = Plot_3D_With_Labels(spon_embedding,spon_label_unsup)
plt.show()
#%% counter 
# np.sum((spon_label_unsup>0)*(spon_label_unsup<9))
spon_events,spon_len = Label_Event_Cutter((spon_label_unsup>0)*(spon_label_unsup>8))
#%% Shuffle 10000 Times, get coactive and all.
spike_series_od = (spon_label_unsup>0)*(spon_label_unsup<9)
spike_series_orien = (spon_label_unsup>0)*(spon_label_unsup>8)
od_events,od_event_len = Label_Event_Cutter(spike_series_od)
orien_events,orien_event_len = Label_Event_Cutter(spike_series_orien)
all_events,all_len = Label_Event_Cutter(spon_label_unsup>0)
# shuffle.
N = 10000# it's a little slow, so don't run 100000 every time.
shuffle_shape = pd.DataFrame(0,index = range(N),columns=['Coactivation','mean_length','std'])
for i in tqdm(range(N)): # cycle shuffle
    # generate random series
    rand_od_series = Random_Series_Generator(9273,od_event_len)
    rand_orien_series = Random_Series_Generator(9273,orien_event_len)
    # get combined series.
    combined_series = (rand_od_series+rand_orien_series)>0
    rand_event,rand_len = Label_Event_Cutter(combined_series)
    shuffle_shape.loc[i,:] = [len(rand_event),rand_len.mean(),rand_len.std()]

# plot distribution of random combine.
plt.switch_backend('webAgg')
ax = sns.histplot(data = shuffle_shape,x = 'Coactivation')
plt.show()
#%% average wait time
all_spike_series = (spon_label_unsup>0)
all_event,all_length = Label_Event_Cutter(all_spike_series)
wait_time = np.zeros(len(all_event)-1)
for i in range(1,len(all_event)):
    before_time = all_event[i-1][0]
    after_time = all_event[i][0]
    c_wait_time = after_time-before_time
    wait_time[i-1] = c_wait_time

param = stats.exponweib.fit(wait_time,floc = 1)# in sequence (exp1, k1, loc1, lam1)
# param = stats.exponweib.fit(wait_time)
plt.switch_backend('webAgg')
x = np.linspace(0, 300, 300)
pdf_fitted = stats.exponweib.pdf(x, *param)
fig, ax = plt.subplots()
ax.hist(wait_time, bins=30, density=True, alpha=1, label='Data')
# plt.plot(x, stats.gamma.pdf(x, a=shape, loc=loc, scale=scale))
ax.plot(x, pdf_fitted, 'r-', label='Fitted')
plt.legend()
plt.show()
stats.kstest(wait_time,'exponweib',args = param)
#%%QQ plot
params = stats.exponweib.fit(wait_time,floc = 1)
plt.switch_backend('webAgg')
fig, ax = plt.subplots()
stats.probplot(wait_time, dist=stats.exponweib,sparams = params, plot=ax)
ax.set_title('QQ plot of data vs. Weibull distribution distribution')
ax.set_xlabel('Theoretical quantiles')
ax.set_ylabel('Sample quantiles')
plt.show()