# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 14:25:22 2022

@author: ZR
"""
#%% Import part

import OS_Tools_Kit as ot
import cv2
from Series_Analyzer.Spontaneous_Preprocessing import Pre_Processor
from Series_Analyzer.Cell_Frame_PCA import Do_PCA,PCA_Regression
import matplotlib.pyplot as plt
from Series_Analyzer.Single_Component_Visualize import Single_Mask_Visualize
from Stimulus_Cell_Processor.Get_Tuning import Get_Tuned_Cells
import scipy.stats as stats
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from Decorators import Timer
from scipy.optimize import curve_fit

#%% calculation part.
def Mises_Function(c_angle,best_angle,a0,b1,b2,c1,c2):
    '''
    Basic orientation fit function. Angle need to be input as RADIUS!!!!
    Parameters see the elife essay.
    '''
    y = a0+b1*np.exp(c1*np.cos(c_angle-best_angle))+b2*np.exp(c2*np.cos(c_angle-best_angle-np.pi))
    return y

class Orientation_Pref_Fit(object):
    
    '''
   A class to fit cell orientation preference. Orientation 0 means up moving horiziontal bars.
    '''
    
    name = r'Orientation Pref fit'
    
    def __init__(self,day_folder,orien_run = 'Run002',run_type = 'G16',used_frame = [4,5]):
        
        # make input variable class variable.
        self.day_folder = day_folder
        print('Make sure cell calculation done before.')
        if run_type == 'G16' or run_type == 'G8' or run_type == 'OD':
            self.run_type = run_type
        else:
            raise IOError('Invalid orientation mode.')
        
        self.used_frame = used_frame
        self.angle_radius_index = 180/np.pi
        
        # get orientation trains here.
        cell_file_name = ot.Get_File_Name(day_folder,'.ac')[0]
        all_cell_dic = ot.Load_Variable(cell_file_name)
        self.acn = list(all_cell_dic.keys())
        self.all_cell_trains = {}
        for i,cc in enumerate(self.acn):
            tc = all_cell_dic[cc]
            c_trains = tc[orien_run]['CR_Train']
            self.all_cell_trains[cc] = c_trains
            

    def ID_Angle_Response(self):
        
        '''
        Get ID-Angle corespond relationship. G8,G16,OD comptable.
        '''
        
        self.ID_angle_dic = {}
        if self.run_type == 'G16':
            for i in range(1,17):
                self.ID_angle_dic[i] = (i-1)*22.5
        elif self.run_type == 'G8':
            for i in range(1,9):	
                self.ID_angle_dic[i] = (i-1)*45
        elif self.run_type == 'OD':
            for i in range(1,9):
                self.ID_angle_dic[i] = (np.ceil(i/2)-1)*45
        
    @Timer
    def Get_Orien_Data(self):
        
        '''
        This function is used to get data into angle-response pandas pattern.
        '''
        
        self.Orientation_Response_Dic = {}
        all_orien_IDs = list(self.ID_angle_dic.keys())
        for i,cc in tqdm(enumerate(self.acn)): # cycle cell
            c_response = self.all_cell_trains[cc]
            cc_frame = pd.DataFrame(columns = ['angle','response'])
            # get cell data into frame format.
            counter = 0
            for j,c_id in enumerate(all_orien_IDs): # cycle ids
                cc_cid_resp = c_response[c_id]
                c_angle = self.ID_angle_dic[c_id]
                for k in range(cc_cid_resp.shape[0]):
                    c_cond_resp = cc_cid_resp[k,self.used_frame].mean()
                    cc_frame.loc[counter] = [c_angle,c_cond_resp]
                    counter +=1
            cc_frame['angle_rad'] = cc_frame['angle']/self.angle_radius_index
            self.Orientation_Response_Dic[cc] = cc_frame
                    

        
        
    def Angle_Fit_Core(self,cc_response,angle_width = 45):
        
        '''
        This function is used to fit single cell angle-response plot.
        '''

        # fit with given condition. Avoid except..
        try:
            parameters, covariance = curve_fit(Mises_Function, cc_response.loc[:,'angle_rad'],cc_response.loc[:,'response'],maxfev=30000)
        except RuntimeError:
            try:
                parameters, covariance = curve_fit(Mises_Function, cc_response.loc[:,'angle_rad'],cc_response.loc[:,'response'],maxfev=50000,p0 = [0,0,0,0,0,0])
            except RuntimeError:
                try:
                    parameters, covariance = curve_fit(Mises_Function, cc_response.loc[:,'angle_rad'],cc_response.loc[:,'response'],maxfev=50000,p0 = [0.2,1,-0.4,-0.2,1,1.5])
                except RuntimeError:
                    return None,None,None,None,None,None
        # fit function using fitted results.
        filled_angle = np.arange(0,2*np.pi,0.01)
        pred_y = Mises_Function(filled_angle,parameters[0],parameters[1],parameters[2],parameters[3],parameters[4],parameters[5])        
        # calculate best angle 
        best_angle_loc = np.where(pred_y == pred_y.max())[0][0]
        best_angle_rad = filled_angle[best_angle_loc]
        best_angle = best_angle_rad*self.angle_radius_index
        # calculate FWHM
        series_length = len(filled_angle)
        angle_width_rad = angle_width/self.angle_radius_index
        angle_range = round(angle_width_rad/0.01)
        max_id = int(best_angle_loc+angle_range)
        min_id = int(best_angle_loc-angle_range)# find peak location.
        if min_id>=0 and max_id<(series_length-1):
            best_peak = pred_y[min_id:max_id]
        elif min_id<0: # meaning best orien~0
            part_a = pred_y[0:max_id]
            part_b = pred_y[min_id+series_length:series_length-1]
            best_peak =  np.concatenate((part_b,part_a),axis = 0)
        elif max_id>(series_length-1): # meaning best orien~2pi
            part_a = pred_y[min_id:series_length-1]
            part_b = pred_y[0:max_id-series_length]
            best_peak =  np.concatenate((part_a,part_b),axis = 0)
        best_reaction = pred_y[best_angle_loc]# minus a0
        half_peak = best_reaction/2
        FWHM = (best_peak>half_peak).sum()*0.01*self.angle_radius_index
        SNR = best_reaction/FWHM
        return best_angle,best_reaction,parameters,covariance,FWHM,SNR
    
    def Angle_Fit_All(self):
        
        '''
        Fit all cells, and plot fit graph & cell tuning property.
        '''
        
        save_folder = self.day_folder+r'\_Orientation_Fit'
        ot.mkdir(save_folder)
        
        self.Cell_Orientation_Tuning = {}
        for i,cc in enumerate(self.acn):
            self.Cell_Orientation_Tuning[cc] = {}
            cc_response = self.Orientation_Response_Dic[cc]
            c_best_angle,c_best_reaction,c_para,c_cov,c_FWHM,c_SNR = self.Angle_Fit_Core(cc_response,angle_width = 45)
            if c_best_angle != None:
                self.Cell_Orientation_Tuning[cc]['Best_Angle'] = c_best_angle
                self.Cell_Orientation_Tuning[cc]['Best_Reaction'] = c_best_reaction
                self.Cell_Orientation_Tuning[cc]['Parameters'] = c_para
                self.Cell_Orientation_Tuning[cc]['Coviriance'] = c_cov
                self.Cell_Orientation_Tuning[cc]['FWHM'] = c_FWHM
                self.Cell_Orientation_Tuning[cc]['SNR'] = c_SNR
                
                filled_angle = np.arange(0,2*np.pi,0.01)
                pred_y = Mises_Function(filled_angle,c_para[0],c_para[1],c_para[2],c_para[3],c_para[4],c_para[5])        
                sns.lineplot(data = cc_response,x = 'angle',y = 'response')
                sns.lineplot(x = filled_angle*180/np.pi,y = pred_y)
                plt.savefig(save_folder+r'\\'+cc+'.png', dpi = 180)
                plt.close()
            else:
                self.Cell_Orientation_Tuning[cc] = None
            

    def One_Key_Fit(self):
        Oft.ID_Angle_Response()
        Oft.Get_Orien_Data()
        Oft.Angle_Fit_All()
        return self.Cell_Orientation_Tuning
            
            
            

#%% test runs
if __name__ == '__main__':
    day_folder = r'G:\Test_Data\2P\210831_L76_2P'
    Oft = Orientation_Pref_Fit(day_folder)
    Oft.ID_Angle_Response()
    Oft.Get_Orien_Data()
    Oft.Angle_Fit_All()
