
'''
This script do caiman to all data need to be processed.
'''

#%%
from My_Wheels.Caiman_API.Precess_Pipeline import Preprocess_Pipeline



#%%
day_folder = r'D:\ZR\_Temp_Data\211009_L76_2P'
pp = Preprocess_Pipeline(day_folder,runlist=[1,2,3,6,7],boulder = (20,20,20,20))
pp.Do_Preprocess()

