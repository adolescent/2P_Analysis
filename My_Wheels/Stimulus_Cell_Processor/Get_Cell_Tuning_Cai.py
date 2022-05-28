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
import numpy as np

class Tuning_Calculator(object):
    
    name = r'Tuning property calculator'
    
    def __init__(self,day_folder,subfolder = '_CAIMAN',
                 od_run = 'Run006',od_type = 'OD_2P',
                 orien_run = 'Run002',orien_type = 'G16_2P',
                 color_run = 'Run007',color_type = 'Hue7Orien4',
                 sig_thres = 0.05,used_frame = [4,5]):
        
        print('Make sure condition data have already been calculated.')
        # get paras 
        if od_run != None:
            self.od_para = Stim_ID_Combiner(od_type)
            self.od_run = od_run
        else:
            print('No OD run.')
            
        if orien_run != None:
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
            print('No Orientation Run')
            
        if color_run != None:
            if color_type == 'Hue7Orien4':
                self.hue_para = Stim_ID_Combiner('Hue7Orien4_Colors')
                self.orien_run = orien_run
            else:
                raise IOError('Invalid hue parameter.')
        else:
            print('No Color run.')
        # load in
        self.sig_thres = 0.05
        self.used_frame = used_frame
        self.all_cond_data = ot.Load_Variable(ot.join(day_folder,subfolder),'Cell_Condition_Response.pkl')
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
            for j,c_orien in enumerate(all_oriens):
                if self.Cell_Tuning_Dic[cc]['Orientation'][c_orien+'_Response']> best_orien_resp:
                    best_orien = c_orien
                    best_orien_resp = self.Cell_Tuning_Dic[cc]['Orientation'][c_orien+'_Response']
                    best_p = self.Cell_Tuning_Dic[cc]['Orientation'][c_orien+'_p']
            if best_p<self.sig_thres:
                self.Cell_Tuning_Dic[cc]['Orien_Preference'] = best_orien
            else:
                self.Cell_Tuning_Dic[cc]['Orien_Preference'] = 'No_Tuning'
    
    
    
    def Fit_Best_Orientation(self):
        pass
            
        
    