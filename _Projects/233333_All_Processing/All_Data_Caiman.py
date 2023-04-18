
'''
This script do caiman to all data need to be processed.
'''

#%%
from My_Wheels.Caiman_API.Precess_Pipeline import Preprocess_Pipeline



#%%
day_folder = r'D:\ZR\_Temp_Data\211009_L76_2P'
pp = Preprocess_Pipeline(day_folder,runlist=[1,2,3,6,7],boulder = (20,20,20,20))
pp.Do_Preprocess()
from Caiman_API.Map_Generators_CAI import One_Key_T_Map
from Stimulus_Cell_Processor.Get_Cell_Tuning_Cai import Tuning_Calculator
One_Key_T_Map(day_folder, 'Run002', 'G16_2P')
One_Key_T_Map(day_folder, 'Run007', 'RGLum4')
Tc = Tuning_Calculator(day_folder,
                               od_run = None,od_type = None,
                               orien_run = 'Run002',orien_type = 'G16_2P',
                               color_run = None,color_type = None)
Cell_Tuning_Dic,Tuning_Property_Cells = Tc.Calculate_Tuning()
            
#%%
day_folder = r'D:\ZR\_Temp_Data\211015_L84_2P'
pp = Preprocess_Pipeline(day_folder,runlist=[1,2,3,6,7,8],boulder = (20,20,20,20),od_run = 'Run008')
pp.Do_Preprocess()
