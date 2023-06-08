
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
import cv2
import copy
from My_Wheels.Cell_Tools.Cell_Visualization import Cell_Weight_Visualization

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
        self.global_avr = cv2.imread(ot.join(self.wp,'Global_Average_cai.tif'))
        self.orien_type = orien_type
        self.od_type = od_type
        self.color_type = color_type
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
            self.colorrun = '1-'+str(1000+color)[1:]
        else:
            self.colorrun = False
        # Generate Z frames as an preprocessing. This is the first process of all works.
        _ = self.Get_Z_Frames()
        _ = self.Get_Cell_Loc()
        
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
    
    def Get_Cell_Loc(self): # This extract cell location alone, save space.
        if hasattr(self,'Cell_Locs'):
            print('Cell_Locations have already been extracted. ')
            return self.Cell_Locs
        self.Cell_Locs = pd.DataFrame(index=['Y','X'],columns=self.acn)
        for i,cc in enumerate(self.acn):
            cc_loc = self.all_cell_dic[cc]['Cell_Loc'].astype('i4')
            self.Cell_Locs[cc] = cc_loc
        return self.Cell_Locs
    
    def Annotate_Cell(self,namelist,on_avr = True,name = True): # This show all cells out. if on_avr == True, stack cell on average graph.
        if on_avr == True:
            annotate_graph = copy.deepcopy(self.global_avr)
        else:
            annotate_graph = np.zeros(shape= (512,512,3),dtype = 'u1')
        for i,cc in enumerate(namelist):
            cc_loc = self.Cell_Locs[cc]
            annotate_graph = cv2.circle(annotate_graph,(cc_loc['X'],cc_loc['Y']),7,(255,0,0),2)
            if name == True:
                annotate_graph = cv2.putText(annotate_graph,str(cc),(cc_loc['X']+7,cc_loc['Y']+7),cv2.FONT_HERSHEY_DUPLEX,0.3,(255,0,0),1,cv2.LINE_AA)
        return annotate_graph
    
    def Generate_Weighted_Cell(self,weight,on_avr = False): # Actually is a direct usage of cell weight visualization
        visualized_graph = Cell_Weight_Visualization(weight,acd = self.all_cell_dic)
        return visualized_graph
    
    def Wash_by_Bright(self,thres_std = 2):
        print(f'Cell Brightness less than {thres_std} std are ignored, others are in variable self.Real_cell.')
        
        return True
    
    def Save_Class(self,path='Default',name='Cell_Class'): # save the class. Get processed datas.
        if path == 'Default':
            ot.Save_Variable(self.wp,name,self)
        else:
            ot.Save_Variable(path,name,self)
    
    
#%% Test parts
if __name__ == '__main__':
    day_folder = r'E:\2P_Raws\220630_L76_2P'
    test_cell = Cell(day_folder)