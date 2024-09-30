'''
This will test the freq and similarity of orientation repeated graph before and after stimulus.
We expect after 100s(130 frame), spon before and after have no difference.

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
from Kill_Cache import kill_all_cache
from sklearn.model_selection import cross_val_score
from sklearn import svm
from scipy.stats import pearsonr
import scipy.stats as stats
from Cell_Class.Plot_Tools import Plot_3D_With_Labels
import copy
from Cell_Class.Advanced_Tools import *
from Classifier_Analyzer import *
from Cell_Class.Timecourse_Analyzer import *


import warnings
warnings.filterwarnings("ignore")

all_path_dic = list(ot.Get_Subfolders(r'D:\_All_Spon_Data_V1'))
all_path_dic.pop(3) # Run 03 bug
all_path_dic.pop(3) # OD bug
all_path_dic.pop(5) # OD bug

save_path = r'D:\_Path_For_Figs\240806_Spon_After'
#%% Get the same model for pca-svm training on orientation.
all_freq_frame = pd.DataFrame(columns = ['Loc','Network','Before_After','Frame_Prop','Freq'])
all_similar_frame = pd.DataFrame(columns = ['Loc','Network','Before_After','Similarity'])
pcnum = 10
fps = 1.301


for i,cloc in tqdm(enumerate(all_path_dic)):
    cloc_name = cloc.split('\\')[-1]
    spon_before = ot.Load_Variable_v2(cloc,'Spon_Before.pkl')
    ac = ot.Load_Variable(cloc,'Cell_Class.pkl')
    spon_after = ac.Z_Frames['1-003'].iloc[500:,:]# ignore first 500 frame.
    series_len = min(len(spon_before),len(spon_after))
    spon_before = spon_before.iloc[:series_len,:]
    spon_after = spon_after.iloc[:series_len,:]
    

    # get before and after network 
    _,_,spon_models = Z_PCA(Z_frame=spon_before,sample='Frame',pcnum=pcnum)
    analyzer_before = Classify_Analyzer(ac = ac,model=spon_models,spon_frame=spon_before,od = 0,orien = 1,color = 0,isi = True)
    analyzer_after = Classify_Analyzer(ac = ac,model=spon_models,spon_frame=spon_after,od = 0,orien = 1,color = 0,isi = True)
    # calculate recover similarity
    analyzer_before.Train_SVM_Classifier(C=1)
    analyzer_before.Similarity_Compare_Average()
    analyzer_after.Train_SVM_Classifier(C=1)
    analyzer_after.Similarity_Compare_Average()

    # get repeat freq and prop of before and after.
    prop_before = np.sum(analyzer_before.spon_label>0)/series_len
    prop_after = np.sum(analyzer_after.spon_label>0)/series_len
    freq_before = Event_Counter(analyzer_before.spon_label>0)*fps/series_len
    freq_after = Event_Counter(analyzer_after.spon_label>0)*fps/series_len

    # and get each network's repeat freq before and after.
    before_similarity = analyzer_before.Avr_Similarity.groupby('Data').get_group('Real Data')
    after_similarity = analyzer_after.Avr_Similarity.groupby('Data').get_group('Real Data')

    # save vars into frame.
    all_freq_frame.loc[len(all_freq_frame),:] = [cloc_name,'Orien','Before',prop_before,freq_before]
    all_freq_frame.loc[len(all_freq_frame),:] = [cloc_name,'Orien','After',prop_after,freq_after]
    # save similarities.
    for j in range(len(before_similarity)):
        all_similar_frame.loc[len(all_similar_frame),:] = [cloc_name,before_similarity['Network'].iloc[j],'Before',before_similarity['PearsonR'].iloc[j]]
        all_similar_frame.loc[len(all_similar_frame),:] = [cloc_name,        after_similarity['Network'].iloc[j],'After',after_similarity['PearsonR'].iloc[j]]

ot.Save_Variable(save_path,'Before_After_Freq',all_freq_frame)
ot.Save_Variable(save_path,'Before_After_Similar',all_similar_frame)
#%% Compare similarity before and after, and freq before and after.
freq_befores = all_freq_frame.groupby('Before_After').get_group('Before')
freq_afters = all_freq_frame.groupby('Before_After').get_group('After')
similar_befores = all_similar_frame.groupby('Before_After').get_group('Before')
similar_afters = all_similar_frame.groupby('Before_After').get_group('After')

#%%
plt.clf()
plt.cla()
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (8,6),dpi = 300)
# sns.barplot(data = all_freq_frame,x = 'Loc',hue = 'Before_After',y = 'Freq',ax = ax)

sns.barplot(data = all_similar_frame,x = 'Loc',hue = 'Before_After',y = 'Similarity',ax = ax)
#%% Compare each loc's F value variation, let's test the hypothesis.
all_F_var = pd.DataFrame(columns = ['Loc','Cell','dF','dF_ratio'])

for i,cloc in tqdm(enumerate(all_path_dic)):
    cloc_name = cloc.split('\\')[-1]
    ac = ot.Load_Variable(cloc,'Cell_Class.pkl')
    cell_num = len(ac)
    for j in tqdm(range(cell_num)):
        cc = j+1
        cc_before = ac[cc]['1-001'].mean()
        cc_after = ac[cc]['1-003'].mean()
        cc_ratio = (cc_after-cc_before)/cc_before
        all_F_var.loc[len(all_F_var),:] = [cloc_name,cc,cc_after-cc_before,cc_ratio]

#%%
plotable = copy.deepcopy(all_F_var)
# plotable['dF_ratio'] = abs(plotable['dF_ratio'])
plt.clf()
plt.cla()
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (8,6),dpi = 300)
# sns.barplot(data = all_freq_frame,x = 'Loc',hue = 'Before_After',y = 'Freq',ax = ax)

sns.boxplot(data = plotable ,x = 'Loc',hue = 'Loc',y = 'dF_ratio',ax = ax,showfliers = False)
ax.axhline(y = 0,color = 'gray',linestyle='--')
# ax.set_ylim(-0.3,0.5)