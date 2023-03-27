
#%%

import cv2
import OS_Tools_Kit as ot
from tqdm import tqdm
from Filters import Signal_Filter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

acd = ot.Load_Variable(r'D:\ZR\_Temp_Data\210920_L76_2P\_CAIMAN\All_Series_Dic.pkl')
cell_num = len(acd)
all_series_dic = np.zeros(shape = (cell_num,14000),dtype = 'f8')


for i in tqdm(range(cell_num)):
    cc = acd[i+1]['1-001'][:14000]
    cc_f = Signal_Filter(cc,filter_para=(0.005*2/1.301,0.3*2/1.301))
    cc_dff = (cc_f-cc_f.mean())/cc_f.mean()
    cc_dff = cc_dff/cc_dff.std()
    all_series_dic[i,:] = cc_dff

#%%
all_series_dic_pos = all_series_dic*(all_series_dic>0)
non_thres_sum = all_series_dic.sum(0)
thres0_sum = all_series_dic_pos.sum(0)
thres1_sum = (all_series_dic_pos*(all_series_dic>1)).sum(0)
thres2_sum = (all_series_dic_pos*(all_series_dic>2)).sum(0)


#%%
def EZPlot(series) -> None:
    plt.switch_backend('webAgg') 
    plt.plot(series)
    plt.show()