
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
day_folder = r'D:\ZR\_Temp_Data\230321_L85_2P'
pp = Preprocess_Pipeline(day_folder,runlist=[11],align_base='1-011',boulder = (20,20,20,20),od_run = None,orien_run=None,color_run='Run011',od_type=None,orien_type=None,color_type='Color7Dir8')
pp.Do_Preprocess()
# Okc = One_Key_Caiman(day_folder,run_lists = [6,8],align_base = '1-006',n_process = 20,use_cuda = False)
# Okc.Motion_Corr_All()
#%%
from Caiman_API.Map_Generators_CAI import One_Key_T_Map
day_folder = r'D:\ZR\_Temp_Data\230321_L85_2P'
One_Key_T_Map(day_folder, 'Run011', run_type = 'Manual',para = {'H-V':[[1,9,17,25,33,5,13,21,29,37],[3,11,19,27,35,7,15,23,31,39]],'A-O':[[2,10,18,26,34,6,14,22,30,38],[4,12,20,28,36,8,16,24,32,40]]})
