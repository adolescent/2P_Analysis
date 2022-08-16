# -*- coding: utf-8 -*-
"""
Created on Sat May 28 11:50:39 2022

@author: adolescent

Get cell tunings from condition dic
"""

import OS_Tools_Kit as ot
from Standard_Parameters.Stim_Name_Tools import Stim_ID_Combiner
from Standard_Parameters.Sub_Graph_Dics import Sub_Dic_Generator
import scipy.stats 
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm
from sklearn.metrics import r2_score


class Tuning_Calculator(object):
    
    name = r'Tuning property calculator'
    
    def __init__(self,day_folder,subfolder = '_CAIMAN',
                 od_run = 'Run006',od_type = 'OD_2P',
                 orien_run = 'Run002',orien_type = 'G16_2P',
                 color_run = 'Run007',color_type = 'HueNOrien4',
                 sig_thres = 0.05,used_frame = [4,5]):
        
        print('Make sure condition data have already been calculated.')
        # get paras 
        if od_run != None:
            self.have_od = True
            self.od_para = Stim_ID_Combiner(od_type)
            self.od_run = od_run
        else:
            self.have_od = False
            print('No OD run.')
            
        if orien_run != None:
            self.have_orien = True
            self.orien_type = orien_type # this might be used during orien fit.
            if orien_type == 'G16_2P':
                self.orien_para = Stim_ID_Combiner('G16_Oriens')
                self.orien_run = orien_run
            elif orien_type == 'G8_2P':
                self.orien_para = Stim_ID_Combiner('G8_2P_Oriens')
                self.orien_run = orien_run
            elif orien_type == 'OD_2P':
                self.orien_para = Stim_ID_Combiner('OD_2P_Oriens')
                self.orien_run = orien_run
            else: 
                raise IOError('Invalid Orientation type.')
        else:
            self.have_orien = False
            print('No Orientation Run')
            
        if color_run != None:
            self.have_color = True
            if color_type == 'HueNOrien4':
                self.hue_para = Stim_ID_Combiner('Hue7Orien4_Colors')
                self.hue_run = color_run
            else:
                raise IOError('Invalid hue parameter.')
        else:
            self.have_color = False
            print('No Color run.')
        # load in
        self.sig_thres = 0.05
        self.used_frame = used_frame
        self.workpath = ot.join(day_folder,subfolder)
        self.all_cond_data = ot.Load_Variable(self.workpath,'Cell_Condition_Response.pkl')
        self.Cell_Tuning_Dic = {}
        self.Tuning_Property_Cells = {}
        self.acn = list(self.all_cond_data.keys())
        for i,cc in enumerate(self.acn):
            self.Cell_Tuning_Dic[cc] = {}
        
    def Condition_Combine_Lite(self,c_condition,id_lists):
        '''
        This small lite is used to combine multiple contitions together.

        Parameters
        ----------
        c_condition : (Dic)
            Dictionary of single cell single run condition dictionary.
        id_lists : (list)
            Condition ID list you want to combine.

        Returns
        -------
        combined_conditions : (np array)
            Array of combined series.

        '''
        combined_conditions = c_condition[id_lists[0]]
        if len(id_lists)>1:
            for i in range(1,len(id_lists)):
                combined_conditions = np.concatenate((combined_conditions,c_condition[id_lists[i]]),axis = 1)

        return combined_conditions
    
    def Get_OD_Tuning(self):
        
        self.Tuning_Property_Cells['LE_Cells'] = []
        self.Tuning_Property_Cells['RE_Cells'] = []
        # Positive value indicate LE, and negative indicate RE.
        for i,cc in enumerate(self.acn):
            cc_cond = self.all_cond_data[cc][self.od_run]
            self.Cell_Tuning_Dic[cc]['OD'] = {}
            LE_ids = self.od_para['L_All']
            RE_ids = self.od_para['R_All']
            LE_conds = self.Condition_Combine_Lite(cc_cond, LE_ids)
            RE_conds = self.Condition_Combine_Lite(cc_cond, RE_ids)
            LE_response = LE_conds[self.used_frame,:].flatten()
            RE_response = RE_conds[self.used_frame,:].flatten()
            t,p = scipy.stats.ttest_ind(LE_response,RE_response,equal_var = False)
            OD_index = np.clip((LE_response.mean()-RE_response.mean())/(LE_response.mean()+RE_response.mean()),-1,1)
            self.Cell_Tuning_Dic[cc]['OD']['t'] = t
            self.Cell_Tuning_Dic[cc]['OD']['p'] = p
            self.Cell_Tuning_Dic[cc]['OD']['Tuning_Index'] = OD_index
            self.Cell_Tuning_Dic[cc]['OD']['LE_Response'] = LE_response.mean()
            self.Cell_Tuning_Dic[cc]['OD']['RE_Response'] = RE_response.mean()
            if p<self.sig_thres:
                if t>0:
                    self.Tuning_Property_Cells['LE_Cells'].append(cc)
                    self.Cell_Tuning_Dic[cc]['OD']['Preference'] = 'LE'
                    self.Cell_Tuning_Dic[cc]['OD_Preference'] = 'LE'
                    
                else:
                    self.Tuning_Property_Cells['RE_Cells'].append(cc)
                    self.Cell_Tuning_Dic[cc]['OD']['Preference'] = 'RE'
                    self.Cell_Tuning_Dic[cc]['OD_Preference'] = 'RE'
            else:
                self.Cell_Tuning_Dic[cc]['OD']['Preference'] = 'No_Tuning'
                self.Cell_Tuning_Dic[cc]['OD_Preference'] = 'No_Tuning'
                
    def Get_Orientation_Tuning(self):
        
        
        for i,cc in enumerate(self.acn):
            self.Cell_Tuning_Dic[cc]['Orientation'] = {}
        all_oriens = list(self.orien_para.keys())
        if 'Blank' in all_oriens:
            all_oriens.remove('Blank')
        # cycle all oriens
        for i,c_orien in enumerate(all_oriens):
            self.Tuning_Property_Cells[c_orien+'_Cells'] = []
            counter_orien = (float(c_orien[5:])+90)%180
            counter_orien_name = 'Orien'+str(counter_orien)
            c_orien_ID = self.orien_para[c_orien]
            counter_orien_ID = self.orien_para[counter_orien_name]
            for j,cc in enumerate(self.acn):
                cc_cond = self.all_cond_data[cc][self.orien_run]
                c_orien_conds = self.Condition_Combine_Lite(cc_cond, c_orien_ID)
                counter_orien_conds = self.Condition_Combine_Lite(cc_cond, counter_orien_ID)
                c_orien_response = c_orien_conds[self.used_frame,:].flatten()
                counter_orien_response = counter_orien_conds[self.used_frame,:].flatten()
                t,p = scipy.stats.ttest_ind(c_orien_response,counter_orien_response,equal_var = False)
                c_orien_index = np.clip((c_orien_response.mean()-counter_orien_response.mean())/(c_orien_response.mean()+counter_orien_response.mean()),-1,1)
                self.Cell_Tuning_Dic[cc]['Orientation'][c_orien+'_t'] = t
                self.Cell_Tuning_Dic[cc]['Orientation'][c_orien+'_p'] = p
                self.Cell_Tuning_Dic[cc]['Orientation'][c_orien+'_Index'] = c_orien_index
                self.Cell_Tuning_Dic[cc]['Orientation'][c_orien+'_Response'] = c_orien_response.mean()
                if p<self.sig_thres and t>0:
                    self.Tuning_Property_Cells[c_orien+'_Cells'].append(cc)
        # Get best orientations.
        for i,cc in enumerate(self.acn):
            best_orien = all_oriens[0]
            best_orien_resp = self.Cell_Tuning_Dic[cc]['Orientation'][all_oriens[0]+'_Response']
            best_p = self.Cell_Tuning_Dic[cc]['Orientation'][all_oriens[0]+'_p']
            for j,c_orien in enumerate(all_oriens):
                if self.Cell_Tuning_Dic[cc]['Orientation'][c_orien+'_Response']> best_orien_resp:
                    best_orien = c_orien
                    best_orien_resp = self.Cell_Tuning_Dic[cc]['Orientation'][c_orien+'_Response']
                    best_p = self.Cell_Tuning_Dic[cc]['Orientation'][c_orien+'_p']
            if best_p<self.sig_thres:
                self.Cell_Tuning_Dic[cc]['Orien_Preference'] = best_orien
            else:
                self.Cell_Tuning_Dic[cc]['Orien_Preference'] = 'No_Tuning'
    
    
    def Mises_Function(self,c_angle,best_angle,a0,b1,b2,c1,c2):
        '''
        Basic orientation fit function. Angle need to be input as RADIUS!!!!
        Parameters see the elife essay.
        '''
        y = a0+b1*np.exp(c1*np.cos(c_angle-best_angle))+b2*np.exp(c2*np.cos(c_angle-best_angle-np.pi))
        return y
    
    
    
    def Fit_Best_Orientation(self,limit =  99999):
        #Initialize
        rad = 180/np.pi
        self.Tuning_Property_Cells['Fitted_Orien'] = {}
        if self.orien_type == 'G16_2P':
            orien_step = 22.5
            all_oriens = {}
            for i in range(16):
                c_name = 'Orien'+str(i*orien_step)
                all_oriens[c_name] = [i+1]
        elif self.orien_type == 'G8_2P':
            orien_step = 45
            all_oriens = {}
            for i in range(8):
                c_name = 'Orien'+str(i*orien_step)
                all_oriens[c_name] = [i+1]
        all_orien_names = list(all_oriens.keys())

        # Cycle all cells.
        for i,cc in tqdm(enumerate(self.acn)):
            # check if this cell have sig orien tuning.
            if self.Cell_Tuning_Dic[cc]['Orien_Preference'] == 'No_Tuning':
                self.Cell_Tuning_Dic[cc]['Fitted_Orien'] = 'No_Tuning'
                self.Tuning_Property_Cells['Fitted_Orien'][cc] = 'No_Tuning'
                
            else:
                
                c_orien_data = self.all_cond_data[cc][self.orien_run]
                # get cc response data.
                cc_response = pd.DataFrame(columns = ['Orien','Orien_Rad','Response'])
                counter = 0
                for j,c_orien in enumerate(all_orien_names):
                    c_angle = float(c_orien[5:])
                    c_angle_rad = c_angle/rad
                    c_id = all_oriens[c_orien]
                    c_conds = self.Condition_Combine_Lite(c_orien_data,c_id)
                    c_resp = c_conds[self.used_frame,:].mean(0).flatten()
                    for k in range(len(c_resp)):
                        cc_response.loc[counter,:] = (c_angle,c_angle_rad,c_resp[k])
                        counter+=1
                # fit data into function.
                cc_response = cc_response.astype(float)
                cc_response = cc_response.groupby('Orien').mean()# average before fiting
                try:
                    parameters, covariance = curve_fit(self.Mises_Function, cc_response.loc[:,'Orien_Rad'],cc_response.loc[:,'Response'],maxfev=30000)
                except RuntimeError:
                    try:
                        parameters, covariance = curve_fit(self.Mises_Function, cc_response.loc[:,'Orien_Rad'],cc_response.loc[:,'Response'],maxfev=50000,p0 = [0,0,0,0,0,0])
                    except RuntimeError:
                        try:
                            parameters, covariance = curve_fit(self.Mises_Function, cc_response.loc[:,'Orien_Rad'],cc_response.loc[:,'Response'],maxfev=50000,p0 = [0.2,1,-0.4,-0.2,1,1.5])
                        except RuntimeError:
                            parameters = None
                            #covarianve = None
                if type(parameters) == np.ndarray:# meaning we have good fit
                    # fit function using fitted results.
                    filled_angle = np.arange(0,2*np.pi,0.01)
                    pred_y = self.Mises_Function(filled_angle,parameters[0],parameters[1],parameters[2],parameters[3],parameters[4],parameters[5])        
                    # calculate best angle 
                    best_angle_loc = np.where(pred_y == pred_y.max())[0][0]
                    best_angle_rad = filled_angle[best_angle_loc]
                    best_angle = (best_angle_rad*rad)%180
                    self.Cell_Tuning_Dic[cc]['Fitted_Orien'] = best_angle
                    self.Cell_Tuning_Dic[cc]['Fit_Parameters'] = parameters
                    self.Tuning_Property_Cells['Fitted_Orien'][cc] = best_angle
# =============================================================================
#                     if abs(best_angle-float(self.Cell_Tuning_Dic[cc]['Orien_Preference'][5:])) < limit*orien_step:# a good fit.
#                         self.Cell_Tuning_Dic[cc]['Fitted_Orien'] = best_angle
#                         self.Tuning_Property_Cells['Fitted_Orien'][cc] = best_angle
#                     else:
#                         print('Angle error exceed, use best tuning.')
#                         self.Cell_Tuning_Dic[cc]['Fitted_Orien'] = float(self.Cell_Tuning_Dic[cc]['Orien_Preference'][5:])
#                         self.Tuning_Property_Cells['Fitted_Orien'][cc] = float(self.Cell_Tuning_Dic[cc]['Orien_Preference'][5:])
# =============================================================================
                    # get r2.
                    #avr_response = cc_response.groupby('Orien').mean()
                    pred_y_r2 = self.Mises_Function(cc_response['Orien_Rad'],parameters[0],parameters[1],parameters[2],parameters[3],parameters[4],parameters[5])
                    r2 = r2_score(cc_response['Response'],pred_y_r2)
                    self.Cell_Tuning_Dic[cc]['Fit_R2'] = r2

                else:
                    print('Fit failed, use best tuning.')
                    self.Cell_Tuning_Dic[cc]['Fitted_Orien'] = float(self.Cell_Tuning_Dic[cc]['Orien_Preference'][5:])
                    self.Cell_Tuning_Dic[cc]['Fit_R2'] = 0
                    self.Tuning_Property_Cells['Fitted_Orien'][cc] = float(self.Cell_Tuning_Dic[cc]['Orien_Preference'][5:])
                # fit function using fitted results.
            
            
    def Get_Hue_Tuning(self):
        
        # Initalize
        all_hues = list(self.hue_para.keys())
        if 'All' in all_hues:
            all_hues.remove('All')
        if 'White' not in all_hues:
            raise IOError('No white condition.')

        white_conds = self.hue_para['White']
        no_white_hues = all_hues.copy()
        no_white_hues.remove('White')
        for i,cc in enumerate(self.acn):
            self.Cell_Tuning_Dic[cc]['Hue'] = {}
        
        # cycle all hues
        for i,c_hue in enumerate(no_white_hues):
            self.Tuning_Property_Cells[c_hue+'_Cells'] = []
            c_hue_id = self.hue_para[c_hue]
            for j,cc in enumerate(self.acn):
                cc_cond = self.all_cond_data[cc][self.hue_run]
                c_hue_conds = self.Condition_Combine_Lite(cc_cond, c_hue_id)
                c_hue_response = c_hue_conds[self.used_frame,:].flatten()
                c_white_conds = self.Condition_Combine_Lite(cc_cond, white_conds)
                c_white_response = c_white_conds[self.used_frame,:].flatten()
                t,p = scipy.stats.ttest_ind(c_hue_response,c_white_response,equal_var = False)
                c_hue_index = np.clip((c_hue_response.mean()-c_white_response.mean())/((c_hue_response.mean()+c_white_response.mean())),-1,1)
                self.Cell_Tuning_Dic[cc]['Hue'][c_hue+'_t'] = t
                self.Cell_Tuning_Dic[cc]['Hue'][c_hue+'_p'] = p
                self.Cell_Tuning_Dic[cc]['Hue'][c_hue+'_Index'] = c_hue_index
                self.Cell_Tuning_Dic[cc]['Hue'][c_hue+'_Response'] = c_hue_response.mean()
                if p<self.sig_thres and t>0:
                    self.Tuning_Property_Cells[c_hue+'_Cells'].append(cc)
        # Get best hues.
        for i,cc in enumerate(self.acn):
            best_hue = no_white_hues[0]
            best_hue_resp = self.Cell_Tuning_Dic[cc]['Hue'][no_white_hues[0]+'_Response']
            best_p = self.Cell_Tuning_Dic[cc]['Hue'][no_white_hues[0]+'_p']
            for j,c_hue in enumerate(no_white_hues):
                if self.Cell_Tuning_Dic[cc]['Hue'][c_hue+'_Response'] > best_hue_resp:
                    best_hue = c_hue
                    best_hue_resp = self.Cell_Tuning_Dic[cc]['Hue'][c_hue+'_Response']
                    best_p = self.Cell_Tuning_Dic[cc]['Hue'][c_hue+'_p']
            if best_p<self.sig_thres:
                self.Cell_Tuning_Dic[cc]['Hue_Preference'] = best_hue
            else:
                self.Cell_Tuning_Dic[cc]['Hue_Preference'] = 'No_Tuning'
    
            
    def Calculate_Tuning(self):
        if self.have_od:
            self.Get_OD_Tuning()
        if self.have_orien:
            self.Get_Orientation_Tuning()
            self.Fit_Best_Orientation()
        if self.have_color:
            self.Get_Hue_Tuning()
            
        ot.Save_Variable(self.workpath, 'Cell_Tuning_Dic', self.Cell_Tuning_Dic)
        ot.Save_Variable(self.workpath, 'Tuning_Property', self.Tuning_Property_Cells)
        
        
        return self.Cell_Tuning_Dic,self.Tuning_Property_Cells
    
#%% Test run
if __name__ == '__main__':
    day_folder = r'D:\ZR\_Temp_Data\220420_L91'
    Tc = Tuning_Calculator(day_folder,od_run = 'Run006',orien_run = 'Run007',color_run = 'Run008')
    Tc.Get_Orientation_Tuning()
    Tc.Fit_Best_Orientation()
    a = Tc.Cell_Tuning_Dic

