
'''

This class will generate item based coding method, we give in data with calculated caiman, all results will be returned.

'''
#%% Imports 
import OS_Tools_Kit as ot
import numpy as np
import pandas as pd
from Filters import Signal_Filter
import matplotlib.pyplot as plt
from Decorators import Timer

#%%
class Cell(object):
    
    name = 'Single Run Cell'
    
    def __init__(self,day_folder,od = 6,orien = 2,color = 7,sub_folder = '_CAIMAN',od_type = 'OD_2P',orien_type = 'G16',color_type = 'Hue7Orien4',filter_para = (0.005,0.3),fps = 1.301,clip_std = 10):
        print('Using Cell Class for data process, make sure data have already been caimaned.')
        self.wp = ot.join(day_folder,sub_folder)
        self.Stim_Frame_Align = ot.Load_Variable(day_folder,'_All_Stim_Frame_Infos.sfa')
        self.all_cell_dic = ot.Load_Variable(self.wp,'All_Series_Dic.pkl')
        self.acn = list(self.all_cell_dic.keys())
        self.cellnum = len(self.acn)
        self.all_run_lists = np.array(list(self.all_cell_dic[1].keys()))[2:]
        self.filter_para = filter_para
        self.fps = fps
        self.clip_std = clip_std
        # Get real id of each run. If False, print and ignore 
        if orien != False:
            self.orienrun = '1-'+str(1000+orien)[1:]
        else:
            self.orienrun == False
        if od != False:
            self.odrun = '1-'+str(1000+od)[1:]
        else:
            self.odrun = False
        if color != False:
            self.color = '1-'+str(1000+color)[1:]
        else:
            self.color = False
        # Generate Z frames as an preprocessing. This is the first process of all works.
        _ = self.Get_Z_Frames()
        
    def __getitem__(self,key):# return a dictionary of specific cell trains. 
        return self.all_cell_dic[key]
    
    def __len__(self):
        return self.cellnum
    
    @Timer
    def Get_Z_Frames(self,mode = 'Z_Score'):# Generate Z scored data of each frame. It is the most important part of this class. All following analysis are based on them.
        if hasattr(self,'Z_Frames'):
            print('Z Score have already been done. ')
            return self.Z_Frames
        self.Z_Frames = {}
        for i,c_run in enumerate(self.all_run_lists):# Cycle of runs. Each run in a dictionary
            frame_num = len(self.all_cell_dic[1][c_run])
            c_frame = pd.DataFrame(columns = self.acn,index = range(frame_num))
            for j,cc in enumerate(self.acn):# cycle of cells. All Cell shall be in the same data frame.
                c_train = self.all_cell_dic[cc][c_run]
                filted_c_train = Signal_Filter(c_train,order = 5,filter_para = (self.filter_para[0]*2/self.fps,self.filter_para[1]*2/self.fps))
                dff_train = (filted_c_train-filted_c_train.mean())/filted_c_train.mean()
                z_train = dff_train/dff_train.std()
                clipped_z_train = np.clip(z_train,-self.clip_std,self.clip_std)
                c_frame[cc] = clipped_z_train
            self.Z_Frames[c_run] = c_frame
        return self.Z_Frames
    
    def Get_Cell_Loc(self):
        pass
    
    def Show_Cell(self,cellname,on_avr = True): # This show all cells out. if on_avr == True, stack cell on average graph.
        pass
    
    def Generate_Weighted_Cell(self,weight,on_avr = False): # weight need to be a pd array with name. Used for graph generation.
        pass
    def Save_Class(self,path='Default',name='Cell_Class'): # save the class. Get processed datas.
        pass
    
    
#%% Test parts
if __name__ == '__main__':
    day_folder = r'E:\2P_Raws\220630_L76_2P'
    cc = Cell(day_folder)