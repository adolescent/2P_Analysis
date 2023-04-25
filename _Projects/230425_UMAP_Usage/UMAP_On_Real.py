'''
Try use UMAP on our data, test each usage.
'''

#%%
import OS_Tools_Kit as ot
import numpy as np
import umap
import umap.plot
import seaborn as sns
import matplotlib.pyplot as plt
from Kill_Cache import kill_all_cache

conda_path = r'C:\ProgramData\anaconda3\envs\umapzr' # if kernel dies, kill them all.
wp = r'D:\ZR\_Data_Temp\_Temp_Data\220711_temp'
cd91 = ot.Load_Variable(wp,'All_Series_Dic91.pkl')
#%% get spon data frame, F value,dF value, dff values(least 10%),dff values(mean),z value.
cell_names = list(cd91.keys())
frame_nums = len(cd91[1]['1-001'])
from Filters import Signal_Filter


