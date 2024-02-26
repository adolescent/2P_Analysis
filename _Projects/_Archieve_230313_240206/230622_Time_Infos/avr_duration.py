'''
This script discript the lasting duration of each repeat event.
Such parameters can be compared with BIG/SMALL ensembles.
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

wp = r'D:\ZR\_Data_Temp\Raw_2P_Data\220420_L91\_CAIMAN'
# wp = r'D:\ZR\_Data_Temp\Raw_2P_Data\220914_L85_2P\_CAIMAN'
ac = ot.Load_Variable(wp,'Cell_Class.pkl')
reducer = ot.Load_Variable(wp,'Stim_All_UMAP_Unsup_3d.pkl')
spon_frame = ac.Z_Frames['1-001']
all_frame,all_label = ac.Combine_Frame_Labels(isi = True)
spon_embeddings = reducer.transform(spon_frame)
stim_embeddings = reducer.transform(all_frame)
#%% embedding check, with origin all stim locations.
plt.switch_backend('webAgg')
fig,ax = Plot_3D_With_Labels(data = stim_embeddings,labels = all_label,ncol=2)
plt.show()
Save_3D_Gif(ax,fig)
#%% do svm with no threshold. this will generate a timeline of repeatition.
classifier,_ = SVM_Classifier(embeddings=stim_embeddings,label = all_label,C = 10)
spon_labels = SVC_Fit(classifier,spon_embeddings,thres_prob=0)
ot.Save_Variable(wp,'spon_svc_labels_0420',spon_labels)
#%% Time infor using spon labels above.
spike_series_od = (spon_labels>0)*(spon_labels<9)
spike_series_orien = (spon_labels>0)*(spon_labels>8)
indices_od = np.where(spike_series_od == 1)[0]
indices_orien = np.where(spike_series_orien == 1)[0]
continuous_series_od = np.split(indices_od, np.where(np.diff(indices_od) != 1)[0]+1)
continuous_series_orien = np.split(indices_orien, np.where(np.diff(indices_orien) != 1)[0]+1)
# get avr length of events.
length_od = np.zeros(len(continuous_series_od))
for i,c_series in enumerate(continuous_series_od):
    length_od[i] = len(c_series)
length_orien = np.zeros(len(continuous_series_orien))
for i,c_series in enumerate(continuous_series_orien):
    length_orien[i] = len(c_series)
#%% Whether this result is inside 
N = 100000
shuffle_shape = pd.DataFrame(0,index = range(N),columns=['Coactivation','mean_length','std'])
for i in tqdm(range(N)): # cycle shuffle
    # generate random OD series
    c_series = np.zeros(11554)
    for j,c_length in enumerate(length_od):
        c_start_loc = np.random.randint(11554-8)
        for k in range(int(c_length)):
            c_series[c_start_loc+k] = 1
    # add random Orientation series on it.
    for j,c_length in enumerate(length_orien):
        c_start_loc = np.random.randint(11554-8)
        for k in range(int(c_length)):
            c_series[c_start_loc+k] = 1
    indices_combine = np.where(c_series == 1)[0]
    continuous_series_combine = np.split(indices_combine, np.where(np.diff(indices_combine) != 1)[0]+1)
    length_combine = np.zeros(len(continuous_series_combine))
    for l,c_series in enumerate(continuous_series_combine):
        length_combine[l] = len(c_series)
    shuffle_shape.loc[i,:] = [len(continuous_series_combine),length_combine.mean(),length_combine.std()]
# Do KS test to see whether the data is inside the distribution.
data = np.array(shuffle_shape['Coactivation'])
x = 610
z = (x-data.mean())/data.std()
stat_value,p = stats.kstest([z], stats.norm.cdf)
print(f'Result in this distribution have p = {p*100:.2f}%')
# plot distribution of random combine.
plt.switch_backend('webAgg')
ax = sns.histplot(data = shuffle_shape,x = 'Coactivation')
plt.show()
#%% Calculate average waiting time of all network repeatance.
all_spike_series = (spon_labels>0)*(spon_labels>0)
def Series_Cutter(input_series):# input series must be 1/0 frame!
    indices_on = np.where(input_series == True)[0]
    cutted_events = np.split(indices_on, np.where(np.diff(indices_on) != 1)[0]+1)
    all_event_length = np.zeros(len(cutted_events))
    for i,c_series in enumerate(cutted_events):
        all_event_length[i] = len(c_series)
    return cutted_events,all_event_length
all_event,all_length = Series_Cutter(all_spike_series)
# get all wait time of events
wait_time = np.zeros(len(all_event)-1)
for i in range(1,len(all_event)):
    before_time = all_event[i-1][0]
    after_time = all_event[i][0]
    c_wait_time = after_time-before_time
    wait_time[i-1] = c_wait_time
# fit data with exponential distribution.
# param = stats.expon.fit(wait_time) # param in sequence location,scale.
param = stats.exponweib.fit(wait_time,floc = 0.5)# in sequence (exp1, k1, loc1, lam1)
plt.switch_backend('webAgg')
x = np.linspace(0, 300, 300)
pdf_fitted = stats.exponweib.pdf(x, *param)
fig, ax = plt.subplots()
ax.hist(wait_time, bins=50, density=True, alpha=1, label='Data')
# plt.plot(x, stats.gamma.pdf(x, a=shape, loc=loc, scale=scale))
ax.plot(x, pdf_fitted, 'r-', label='Fitted')
plt.legend()
plt.show()
stats.kstest(wait_time,'exponweib',args = param)
#%% QQPlot
a = copy.deepcopy(wait_time)
# a[a>500]=500
params = stats.exponweib.fit(a,floc = 1)
plt.switch_backend('webAgg')
fig, ax = plt.subplots()
probplot(a, dist=stats.exponweib,sparams = params, plot=ax)
ax.set_title('QQ plot of data vs. exponential distribution')
ax.set_xlabel('Theoretical quantiles')
ax.set_ylabel('Sample quantiles')
plt.show()

#%% see single cell spike distribution.
# dff first.
from scipy.stats import poisson
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
counts = np.array((spon_frame>1).mean(1))
plt.switch_backend('webAgg')
plt.hist(counts,bins = 50)
plt.show()