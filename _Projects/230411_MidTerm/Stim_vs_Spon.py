'''
Compare stim dF/F with spon series.
'''

#%%

from Series_Analyzer.Preprocessor_Cai import Pre_Processor_Cai
from scipy.signal import find_peaks
from Graph_Operation_Kit import EZPlot
import matplotlib.pyplot as plt
import numpy as np
import OS_Tools_Kit as ot
import pandas as pd
import seaborn as sns
import random
from tqdm import tqdm
from scipy.stats import pearsonr
from scipy.stats import ttest_rel

def LinePlot(series) -> None:
    plt.switch_backend('webAgg') 
    plt.plot(series)
    plt.show()
day_folder = r'F:\_Data_Temp\220630_L76_2P'


run01_frame = Pre_Processor_Cai(day_folder,'Run001',base_mode='most_unactive',use_z = False)
run07_frame = Pre_Processor_Cai(day_folder,'Run007',base_mode='most_unactive',use_z= False)
#%%


cell_num = run01_frame.shape[0]
# compare cell num
max_dff = pd.DataFrame(index = range(1,cell_num+1),columns=['Spon','Stim'])
for i in range(cell_num):
    series_spon = run01_frame.iloc[i,:-300].reset_index(drop=True)
    series_stim = run07_frame.iloc[i,:]
    max_dff.loc[i+1,:]=[series_spon.max(),series_stim.max()]
#%% find and get peak distribution.
tc = 47 # in iloc
xs = run01_frame.iloc[tc,6000:8000].reset_index(drop = True)
x = run07_frame.iloc[tc,:2000].reset_index(drop = True)
peaks, properties = find_peaks(x, height=0.2,distance = 3,width = 0)
plt.switch_backend('webAgg') 
plt.plot(x)
plt.plot(xs)
# plt.plot(np.zeros_like(x), "--", color="gray")
# plt.plot(peaks, x[peaks], "x")
plt.show()
#%% Count total dF/F peak in 3000 frames.
total_dff = pd.DataFrame(index = range(1,cell_num+1),columns = ['Stim','Spon'])
for i in tqdm(range(cell_num)):
    c_spon_series = run01_frame.iloc[i,6000:8000].reset_index(drop = True)
    c_stim_series = run07_frame.iloc[i,:2000].reset_index(drop = True)
    spon_peaks, spon_properties = find_peaks(c_spon_series, height=1.3,distance = 3,width = 0)
    stim_peaks, stim_properties = find_peaks(c_stim_series, height=1.3,distance = 3,width = 0)
    stim_peak_sum = stim_properties['peak_heights'].sum()
    spon_peak_sum = spon_properties['peak_heights'].sum()
    total_dff.iloc[i,:] = [stim_peak_sum,spon_peak_sum]


#%% need to be done manually
# stim_peak_heights = properties['peak_heights'].copy()
# spon_peak_heights = properties['peak_heights'].copy()
# #%% plot hist graph.
# plt.switch_backend('webAgg') 
# plt.hist(spon_peak_heights,bins = 15,alpha = 0.7)
# plt.hist(stim_peak_heights,bins = 15,alpha = 0.7)
# plt.show()
#%% plot line graph
plt.switch_backend('webAgg') 
g = sns.scatterplot(data = total_dff,x = 'Spon',y = 'Stim',s = 7)
# sns.lineplot(x = range(1,400),y = range(1,400),color = 'r')
# g.set(ylim = (100,250))
# g.set(xlim = (100,250))
plt.show()

ttest_rel(total_dff['Spon'],total_dff['Stim'])

#%%
# plt.cla()
# plt.clf()
# plt.scatter(series_spon)
# plt.scatter(series_stim)
plt.switch_backend('webAgg') 
# plt.switch_backend('QtAgg') 
plt.figure(figsize = (7,7))
g = sns.scatterplot(data = max_dff,x = 'Spon',y = 'Stim',s = 6)
sns.lineplot(x = range(0,10),y = range(0,10),color =  'r')
g.set(ylim = (0,6))
g.set(xlim = (0,6))
plt.show()
ttest_rel(max_dff['Spon'],max_dff['Stim'])
#%%
plt.clf()
# plt.scatter(series_spon)
# plt.scatter(series_stim)
plt.hist(max_dff['Spon'])
plt.hist(max_dff['Stim'])
plt.show()

#%% Compare stim ISI and spon dFF.
