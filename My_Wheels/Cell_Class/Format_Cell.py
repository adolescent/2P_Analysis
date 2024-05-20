
'''

This class will generate item based coding method, we give in data with calculated caiman, all results will be returned.

'''
#%% Imports 
import OS_Tools_Kit as ot
import numpy as np
import pandas as pd
from Filters import Signal_Filter,Signal_Filter_v2
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
            self.orienrun = False
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
        self.Regenerate_Cell_Graph()
        
    def __getitem__(self,key):# return a dictionary of specific cell trains. 
        return self.all_cell_dic[key]
    
    def __len__(self):
        return self.cellnum
    
    @Timer
    def Get_Z_Frames(self):# Generate Z scored data of each frame. It is the most important part of this class. All following analysis are based on them.
        if hasattr(self,'Z_Frames'):
            print('Z Score have already been done. ')
            return self.Z_Frames
        self.Z_Frames = {}
        for i,c_run in enumerate(self.all_run_lists):# Cycle of runs. Each run in a dictionary
            frame_num = len(self.all_cell_dic[1][c_run])
            c_frame = pd.DataFrame(columns = self.acn,index = range(frame_num))
            for j,cc in enumerate(self.acn):# cycle of cells. All Cell shall be in the same data frame.
                c_train = self.all_cell_dic[cc][c_run]
                # filted_c_train = Signal_Filter(c_train,order = 5,filter_para = (self.filter_para[0]*2/self.fps,self.filter_para[1]*2/self.fps))
                filted_c_train = Signal_Filter_v2(series = c_train,HP_freq=self.filter_para[0],LP_freq=self.filter_para[1],fps = self.fps)
                dff_train = (filted_c_train-filted_c_train.mean())/filted_c_train.mean()
                z_train = dff_train/dff_train.std()
                clipped_z_train = np.clip(z_train,-self.clip_std,self.clip_std)
                c_frame[cc] = clipped_z_train
            self.Z_Frames[c_run] = c_frame
        return self.Z_Frames
    
    def Get_dFF_Frames(self,runname,prop = 0.1,start = 0,stop = 99999):
        # get f matrix first.
        real_stop = min(len(self.all_cell_dic[1][runname]),stop)
        F_frames_all = np.zeros(shape = (real_stop-start,len(self.acn)),dtype='f8')
        for i,cc in enumerate(self.acn):
            c_series_raw = self.all_cell_dic[cc][runname][start:real_stop]
            c_series_filted = Signal_Filter_v2(c_series_raw,order=5,HP_freq=self.filter_para[0],LP_freq=self.filter_para[1],fps = self.fps)
            F_frames_all[:,i] = c_series_filted
        # then calculate dF/F matrix.
        dFF_matrix = np.zeros(shape = F_frames_all.shape,dtype='f8')
        for i in range(F_frames_all.shape[1]):
            c_F_series = F_frames_all[:,i]
            base_num = int(len(c_F_series)*prop)
            base_id = np.argpartition(c_F_series, base_num)[:base_num]
            base = c_F_series[base_id].mean()
            c_dff_series = (c_F_series-base)/base
            dFF_matrix[:,i] = c_dff_series
        return dFF_matrix

    
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
    
    def Generate_Weighted_Cell(self,weight): # Actually is a direct usage of cell weight visualization
        visualized_graph = Cell_Weight_Visualization(weight,acd = self.all_cell_dic)
        return visualized_graph
    
    def Regenerate_Cell_Graph(self,clip_std = 10,gain = 2.5):
        # This will regenerate cell bright graph. 
        # get frame weight of all cells.
        runlists = self.Z_Frames.keys()
        run_num = len(runlists)
        frame_num = np.zeros(run_num)
        for i,c_run in enumerate(runlists):
            frame_num[i] = self.Z_Frames[c_run].shape[0]
        frame_num_all = frame_num.sum()
        # calculate cellular 
        cells_new = np.zeros(shape = (512,512),dtype = 'f8')
        self.all_cell_F = np.zeros(self.cellnum)
        for i,cc in enumerate(self.acn):
            F_avr = 0
            for j,c_run in enumerate(runlists):
                F_avr += self.all_cell_dic[cc][c_run].mean()*frame_num[j]
            F_avr = F_avr/frame_num_all
            self.all_cell_F[i] = F_avr
            cells_new = cells_new+self.all_cell_dic[cc]['Cell_Mask']*self.all_cell_F[i]
            
        cells_new = np.clip(cells_new,0,cells_new+cells_new.std()*clip_std)
        self.new_avr_graph = (np.clip((cells_new/cells_new.max())*gain,0,1)*255).astype('u1')
        cv2.imwrite(ot.join(self.wp,'New_All_Cells.png'),self.new_avr_graph)
    
    def Wash_by_Bright(self,thres_std = 1):
        print(f'Cell Brightness less than {thres_std} std are ignored, others are in variable self.Real_cell.')
        
        # get threshold of F value.
        F_thres = self.all_cell_F.mean()-self.all_cell_F.std()
        cell_flag = self.all_cell_F>F_thres
        # Get real cell list.
        self.passed_cells = []
        cells_after_wash = np.zeros(shape = (512,512),dtype = 'f8')
        for i,cc in enumerate(self.acn):
            if cell_flag[i]==True:
                self.passed_cells.append(cc)
                cc_mask = self.all_cell_dic[cc]['Cell_Mask']
                cells_after_wash = cells_after_wash+(cc_mask*255*30)
        cells_after_wash.astype('u1')
        cv2.imwrite(ot.join(self.wp,'Cells_After_Wash.png'),cells_after_wash)
        return self.passed_cells,cells_after_wash
    
    def Save_Class(self,path='Default',name='Cell_Class'): # save the class. Get processed datas.
        if path == 'Default':
            ot.Save_Variable(self.wp,name,self)
        else:
            ot.Save_Variable(path,name,self)
    
    
#%% Test parts
if __name__ == '__main__':
    day_folder = r'D:\ZR\_Data_Temp\Raw_2P_Data\220902_L76_2P'
    test_cell = Cell(day_folder)
    test_cell.Regenerate_Cell_Graph(clip_std = 2.5)