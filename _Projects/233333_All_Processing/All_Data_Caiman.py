
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
#%%
day_folder = r'D:\ZR\_Data_Temp\220111_L76_2P'
pp = Preprocess_Pipeline(day_folder,runlist=[1,2,3,6,7],boulder = (20,20,20,20))
pp.Do_Preprocess()
#%%
day_folder = r'D:\ZR\_Data_Temp\220211_L76_2P'
pp = Preprocess_Pipeline(day_folder,runlist=[1,2,3,4,5],boulder = (20,20,20,20),od_run = None,orien_run = 'Run002',color_run = 'Run005')
pp.Do_Preprocess()
#%%
day_folder = r'D:\ZR\_Data_Temp\220304_L76_2P'
pp = Preprocess_Pipeline(day_folder,runlist=[1,3,4,7,8],boulder = (20,50,20,20),orien_run = 'Run003',od_run = 'Run007',color_run = 'Run008')
pp.Do_Preprocess()
#%%
day_folder = r'D:\ZR\_Data_Temp\220310_L85'
pp = Preprocess_Pipeline(day_folder,runlist=[1,2,3,6,7,8],boulder = (20,50,75,20),orien_run = 'Run002',od_run = 'Run006',color_run = 'Run007')
pp.Do_Preprocess()

#%%
day_folder = r'D:\ZR\_Data_Temp\220324_L85'
pp = Preprocess_Pipeline(day_folder,runlist=[1,2,3,6,7],boulder = (20,20,40,20),orien_run = 'Run002',od_run = 'Run006',color_run = 'Run007')
pp.Do_Preprocess()

#%%
day_folder = r'D:\ZR\_Data_Temp\220326_L76'
pp = Preprocess_Pipeline(day_folder,runlist=[1,2,3,6,7],boulder = (20,20,20,20),orien_run = 'Run002',od_run = 'Run006',color_run = 'Run007')
pp.Do_Preprocess()
#%%
day_folder = r'D:\ZR\_Data_Temp\220405_L91'
pp = Preprocess_Pipeline(day_folder,runlist=[2,3,4,7,8],boulder = (20,20,20,20),orien_run = 'Run003',od_run = 'Run007',color_run = 'Run008')
pp.Do_Preprocess()
#%%
day_folder = r'D:\ZR\_Data_Temp\220407_L85'
pp = Preprocess_Pipeline(day_folder,runlist=[1,2,3,6,7],boulder = (20,20,20,20),orien_run = 'Run002',od_run = 'Run006',color_run = 'Run007')
pp.Do_Preprocess()
#%%
day_folder = r'D:\ZR\_Data_Temp\220415_L76'
pp = Preprocess_Pipeline(day_folder,runlist=[1,2,3,6,7,8],boulder = (20,20,20,20),orien_run = 'Run007',od_run = 'Run006',color_run = 'Run008')
pp.Do_Preprocess()
#%%
day_folder = r'D:\ZR\_Data_Temp\220420_L91'
pp = Preprocess_Pipeline(day_folder,runlist=[1,2,3,6,7,8],boulder = (20,20,20,20),orien_run = 'Run007',od_run = 'Run006',color_run = 'Run008')
pp.Do_Preprocess()
#%%
day_folder = r'D:\ZR\_Data_Temp\220504_L91'
pp = Preprocess_Pipeline(day_folder,runlist=[1,2,3,6,7],boulder = (20,20,20,75),orien_run = 'Run002',od_run = 'Run006',color_run = 'Run007',align_base = '1-002',max_shift = (120,75))
pp.Do_Preprocess()
#%%
day_folder = r'D:\ZR\_Data_Temp\220505_L85'
pp = Preprocess_Pipeline(day_folder,runlist=[1,2,3,6,7],boulder = (20,20,20,20),orien_run = 'Run002',od_run = 'Run006',color_run = 'Run007',align_base = '1-002',max_shift = (75,75))
pp.Do_Preprocess()

#%%
day_folder = r'D:\ZR\_Data_Temp\220506_L76_2P'
pp = Preprocess_Pipeline(day_folder,runlist=[1,2,3,6,7],boulder = (20,20,20,20),orien_run = 'Run002',od_run = 'Run006',color_run = 'Run007',align_base = '1-002',max_shift = (75,75))
pp.Do_Preprocess()
#%%
day_folder = r'D:\ZR\_Data_Temp\220608_L85_2P'
pp = Preprocess_Pipeline(day_folder,runlist=[2,3,4,7,8,9],boulder = (20,20,20,20),orien_run = 'Run008',od_run = 'Run007',color_run = 'Run009',align_base = '1-003',max_shift = (75,75))
pp.Do_Preprocess()
#%%
day_folder = r'D:\ZR\_Data_Temp\220609_L91_2P'
pp = Preprocess_Pipeline(day_folder,runlist=[1,2,3,6,7,8],boulder = (20,20,20,20),orien_run = 'Run007',od_run = 'Run006',color_run = 'Run008',align_base = '1-003',max_shift = (75,75))
pp.Do_Preprocess()
#%%
day_folder = r'D:\ZR\_Data_Temp\220630_L76_2P'
pp = Preprocess_Pipeline(day_folder,runlist=[1,3,6,7,8],boulder = (20,20,20,20),orien_run = 'Run007',od_run = 'Run006',color_run = 'Run008',align_base = '1-003',max_shift = (75,75))
pp.Do_Preprocess()
#%%
day_folder = r'D:\ZR\_Data_Temp\220706_L85_LM'
pp = Preprocess_Pipeline(day_folder,runlist=[1,3,6,7,8],boulder = (20,20,20,35),orien_run = 'Run007',od_run = 'Run006',color_run = 'Run008',align_base = '1-003',max_shift = (75,75))
pp.Do_Preprocess()
#%%
day_folder = r'D:\ZR\_Data_Temp\220712_L76_2P'
pp = Preprocess_Pipeline(day_folder,runlist=[1,3,6,7,8],boulder = (20,20,20,35),orien_run = 'Run007',od_run = 'Run006',color_run = 'Run008',align_base = '1-003',max_shift = (75,75))
pp.Do_Preprocess()
#%%
day_folder = r'D:\ZR\_Data_Temp\220713_L85_2P'
pp = Preprocess_Pipeline(day_folder,runlist=[1,2,3,6,7],boulder = (20,20,20,35),orien_run = 'Run002',od_run = 'Run006',color_run = 'Run007',align_base = '1-002',max_shift = (75,75))
pp.Do_Preprocess()
#%%
day_folder = r'D:\ZR\_Data_Temp\220721_L76_2P'
pp = Preprocess_Pipeline(day_folder,runlist=[1,2,3,6,7,8],boulder = (20,20,20,35),orien_run = 'Run007',od_run = 'Run006',color_run = 'Run008',align_base = '1-003',max_shift = (75,75))
pp.Do_Preprocess()
#%%
day_folder = r'D:\ZR\_Data_Temp\220727_L85_2P'
pp = Preprocess_Pipeline(day_folder,runlist=[1,2,3,6,7,8],boulder = (20,20,20,20),orien_run = 'Run007',od_run = 'Run006',color_run = 'Run008',align_base = '1-003',max_shift = (75,75))
pp.Do_Preprocess()
#%%
day_folder = r'D:\ZR\_Data_Temp\220810_L85_2P'
pp = Preprocess_Pipeline(day_folder,runlist=[1,2,3,6,7],boulder = (20,35,20,20),orien_run = 'Run002',od_run = 'Run006',color_run = 'Run007',align_base = '1-002',max_shift = (75,75))
pp.Do_Preprocess()
#%%
day_folder = r'D:\ZR\_Data_Temp\220812_L76_2P'
pp = Preprocess_Pipeline(day_folder,runlist=[1,2,3,6,7],boulder = (20,20,20,20),orien_run = 'Run002',od_run = 'Run006',color_run = 'Run007',align_base = '1-002',max_shift = (75,75))
pp.Do_Preprocess()
#%%
day_folder = r'D:\ZR\_Data_Temp\220825_L85_2P'
pp = Preprocess_Pipeline(day_folder,runlist=[1,2,3,6,7],boulder = (20,20,20,20),orien_run = 'Run002',od_run = 'Run006',color_run = 'Run007',align_base = '1-002',max_shift = (75,75))
pp.Do_Preprocess()
#%%
day_folder = r'D:\ZR\_Data_Temp\220902_L76_2P'
pp = Preprocess_Pipeline(day_folder,runlist=[1,2,3,6,7,8],boulder = (20,20,20,20),orien_run = 'Run007',od_run = 'Run006',color_run = 'Run008',align_base = '1-003',max_shift = (75,75))
pp.Do_Preprocess()
#%%
day_folder = r'D:\ZR\_Data_Temp\220914_L85_2P'
pp = Preprocess_Pipeline(day_folder,runlist=[1,2,3,6,7,8],boulder = (20,20,20,20),orien_run = 'Run007',od_run = 'Run006',color_run = 'Run008',align_base = '1-003',max_shift = (75,75))
pp.Do_Preprocess()
#%%
day_folder = r'D:\ZR\_Data_Temp\220421_L85'
pp = Preprocess_Pipeline(day_folder,runlist=[1,2,3,7,8,9],boulder = (20,20,20,20),orien_run = 'Run008',od_run = 'Run007',color_run = 'Run009',align_base = '1-003',max_shift = (75,75))
pp.Do_Preprocess()