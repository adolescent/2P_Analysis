
'''
This script do caiman to all data need to be processed.
'''
#%%
from My_Wheels.Caiman_API.Precess_Pipeline import Preprocess_Pipeline
from My_Wheels.Caiman_API.One_Key_Caiman import One_Key_Caiman


#%%
day_folder = r'D:\ZR\_Temp_Data\210831_L76_2P'
pp = Preprocess_Pipeline(day_folder,runlist=[1,2,3,6,7],boulder = (20,20,20,20))
pp.Do_Preprocess()
#%%
day_folder = r'D:\ZR\_Temp_Data\230321_L85_2P'
pp = Preprocess_Pipeline(day_folder,runlist=[11],align_base='1-011',boulder = (20,20,20,20),od_run = None,orien_run=None,color_run='Run011',od_type=None,orien_type=None,color_type='Color7Dir8')
pp.Do_Preprocess()
# Okc = One_Key_Caiman(day_folder,run_lists = [11],align_base = '1-011')
# Okc.Motion_Corr_All()
# print('Cell_Finding')
# Okc.Cell_Find(boulders= [20,20,20,20])
# print('Series_Generating...')
# Okc.Series_Generator_NG()
#%%
day_folder = r'D:\ZR\_Temp_Data\222222_L76_Test'
# pp = Preprocess_Pipeline(day_folder,runlist=[3,11],align_base='1-011',boulder = (20,20,20,20),od_run = None,orien_run='Run003',color_run='Run011',od_type=None,orien_type='G8+90',color_type='Color7Dir8')
# pp.Do_Preprocess()
Okc = One_Key_Caiman(day_folder,run_lists = [4,5,6],align_base = '1-004',n_process = 20,use_cuda = False)
Okc.Motion_Corr_All()