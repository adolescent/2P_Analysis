'''
This script will generate graph between ensemble size and tuning property.
Use sum of tuning as tuning pref.
'''
#%% import and data reading.
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

def LinePlot(series) -> None:
    plt.switch_backend('webAgg') 
    plt.plot(series)
    plt.show()
day_folder = r'D:\ZR\_Temp_Data\220630_L76_2P'
run01_frames = Pre_Processor_Cai(day_folder)
spikes = run01_frames*(run01_frames>2)
#%% count number of cells in run01
raster_trains = run01_frames>2
frame_counts = raster_trains.sum(0)
# find peaks of cell num >10
x = frame_counts
peaks, properties = find_peaks(x, height=10,distance = 3,width = 0)
plt.switch_backend('webAgg') 
plt.plot(x)
plt.plot(np.zeros_like(x), "--", color="gray")
plt.plot(peaks, x[peaks], "x")
plt.show()
tune_cd = ot.Load_Variable(r'D:\ZR\_Temp_Data\220630_L76_2P\_CAIMAN\Cell_Tuning_Dic.pkl')
tune_info = ot.Load_Variable(r'D:\ZR\_Temp_Data\220630_L76_2P\_CAIMAN\Tuning_Property.pkl')
# get each peak frame from all peak info.
all_peak_frame = spikes.loc[:,peaks]
#%% get a tuning calculator.
def Tuning_Counter(cell_lists,tune_cd,show = False):
    LE_num = 0
    RE_num = 0
    Orien0_num = 0
    Orien45_num = 0
    Orien90_num = 0
    Orien135_num = 0
    for i,cc in enumerate(cell_lists):
        ct = tune_cd[cc]
        if ct['OD']['Tuning_Index']>0.5:
            LE_num +=1
        elif ct['OD']['Tuning_Index']<-0.5:
            RE_num +=1
        if ct['Fitted_Orien'] != 'No_Tuning':
            orien_flag = (ct['Fitted_Orien']+22.5)//45
            if orien_flag ==0:
                Orien0_num +=1
            elif orien_flag ==1:
                Orien45_num +=1
            elif orien_flag ==2:
                Orien90_num +=1
            elif orien_flag ==3:
                Orien135_num +=1
    if show == True:
        print(f'LE cell num: {LE_num},RE cell num: {RE_num}')
        print(f'0 cell num: {Orien0_num},45 cell num: {Orien45_num}')
        print(f'90 cell num: {Orien90_num},135 cell num: {Orien135_num}')
    return LE_num,RE_num,Orien0_num,Orien45_num,Orien90_num,Orien135_num
All_LE,All_RE,All_0,All_45,All_90,All_135 = Tuning_Counter(list(range(1,509)),tune_cd)
#%%
prop_info = pd.DataFrame(index = list(range(len(peaks))),columns = ['Size','LE_prop','RE_prop','0_prop','45_prop','90_prop','135_prop','Orien_pref'])
for i in range(len(peaks)):
    c_frame = all_peak_frame.iloc[:,i]
    c_response = c_frame[c_frame>0]
    cc_list = list(c_response.index)
    c_LE,c_RE,c_0,c_45,c_90,c_135 = Tuning_Counter(cc_list,tune_cd)
    
    all_orien = np.array([c_0/All_0,c_45/All_45,c_90/All_90,c_135/All_135])
    orien_pref = (all_orien.max()-all_orien.mean())/all_orien.mean()
    prop_info.loc[i,:] = [len(cc_list),c_LE/All_LE,c_RE/All_RE,c_0/All_0,c_45/All_45,c_90/All_90,c_135/All_135,orien_pref]

prop_info['OD_index'] = abs((prop_info['LE_prop']-prop_info['RE_prop'])/(prop_info['LE_prop']+prop_info['RE_prop']))

#%% Compare with random select.
times = 1000
ac_list = list(range(1,509))
rand_info = pd.DataFrame(index = range(10,300),columns = ['Act_Num','OD_diff','Orien_diff'])
for i in tqdm(range(10,300)):
    od_pref_list = np.zeros(times)
    orien_pref_list = np.zeros(times)
    for j in range(times):
        cc_list = random.sample(ac_list,i)
        c_LE,c_RE,c_0,c_45,c_90,c_135 = Tuning_Counter(cc_list,tune_cd)
        all_orien = np.array([c_0/All_0,c_45/All_45,c_90/All_90,c_135/All_135])
        LE_prop = c_LE/All_LE
        RE_prop = c_RE/All_RE
        orien_pref = (all_orien.max()-all_orien.mean())/all_orien.mean()
        od_pref = abs((LE_prop-RE_prop)/(LE_prop+RE_prop))
        od_pref_list[j] = od_pref
        orien_pref_list[j] = orien_pref
    rand_info.loc[i] = [i,od_pref_list.mean(),orien_pref_list.mean()]

#%%
plt.clf()
sns.scatterplot(data = prop_info,x = 'Size',y = 'OD_index',s = 5)
# sns.kdeplot(data = prop_info,x = 'Size',y = 'OD_index',fill = True,thresh = 0,levels =100,cmap = 'mako')
sns.lineplot(data = rand_info,x = 'Act_Num',y = 'OD_diff')
plt.show()
#%% group real data in size = 10.
prop_info['Size_Group'] = (prop_info['Size']//5)*5
prop_avr = prop_info.groupby('Size_Group').mean()
plt.clf()
# sns.scatterplot(data = prop_info,x = 'Size',y = 'OD_index',s = 5)
sns.kdeplot(data = prop_info,x = 'Size',y = 'Orien_pref',fill = True,thresh = 0.03,levels =100,cmap = 'mako')
sns.lineplot(data = prop_avr,x = 'Size_Group',y = 'Orien_pref')
sns.lineplot(data = rand_info,x = 'Act_Num',y = 'Orien_diff')
plt.show()
diff = (prop_avr['Orien_pref']-rand_info['Orien_diff']).dropna()
od_diff = (prop_avr['OD_index']-rand_info['OD_diff']).dropna()

