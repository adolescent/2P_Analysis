'''
Regenerate corr vs dist map. plot in kde style.
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

def LinePlot(series) -> None:
    plt.switch_backend('webAgg') 
    plt.plot(series)
    plt.show()
day_folder = r'D:\ZR\_Temp_Data\220630_L76_2P'
run01_frames = Pre_Processor_Cai(day_folder)
cell_num = run01_frames.shape[0]
acd = ot.Load_Variable(r'D:\ZR\_Temp_Data\220630_L76_2P\_CAIMAN\All_Series_Dic.pkl')
#%% Do pair corr.
all_corr = pd.DataFrame(columns = ['Dist','Corr'],index = range(int(507*508/2)))
count = 0
for i in tqdm(range(cell_num)):
    series_A = run01_frames.iloc[i,:]
    A_loc = acd[i+1]['Cell_Loc']
    for j in range(i+1,cell_num):
        series_B = run01_frames.iloc[j,:]
        B_loc = acd[j+1]['Cell_Loc']
        c_dist = np.sqrt(np.power((A_loc[0]-B_loc[0]),2)+np.power((A_loc[1]-B_loc[1]),2))
        c_corr,_ = pearsonr(series_A,series_B)
        all_corr.loc[count,:] = [c_dist,c_corr]
        count += 1
#%%
plt.clf()
# sns.scatterplot(data = prop_info,x = 'Size',y = 'OD_index',s = 5)
sns.kdeplot(data = all_corr,x = 'Dist',y = 'Corr',fill = True,thresh = 0.03,levels =100,cmap = 'rocket')
plt.show()

#%% fit with curve function.
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
def Reci_Func(dist,const,slope,bias):
    corr = const+slope*(1/(dist+bias))
    return corr
# fit data
para, covar = curve_fit(Reci_Func,xdata = all_corr['Dist'],ydata = all_corr['Corr'])
y_pred = Reci_Func(all_corr['Dist'], *para)
r2 = r2_score(all_corr['Corr'], y_pred)
dist_range = np.array(range(1,620))
plt.clf()
sns.kdeplot(data = all_corr,x = 'Dist',y = 'Corr',fill = True,thresh = 0,levels =100,cmap = 'rocket')
plt.plot(dist_range,Reci_Func(dist_range,*para),color = 'y')
plt.show()