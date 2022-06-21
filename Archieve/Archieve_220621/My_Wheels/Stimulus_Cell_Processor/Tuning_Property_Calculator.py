# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 14:14:49 2021

@author: ZR
"""

from Cell_Processor import Cell_Processor
from Standard_Parameters.Stim_Name_Tools import Tuning_IDs
import OS_Tools_Kit as ot
from Stimulus_Cell_Processor.Tuning_Selector import Get_Tuning_Checklists

def Tuning_Property_Calculator(day_folder,
                               Orien_para = ('Run002','G16_2P'),
                               OD_para = ('Run006','OD_2P'),
                               Hue_para = ('Run007','HueNOrien4',{'Hue':['Red','Yellow','Green','Cyan','Blue','Purple','White']}),
                               used_frame = [4,5],sig_thres = 0.05
                               ):
    '''
    Calculate cell tuning property through all cell data.

    Parameters
    ----------
    day_folder : (str)
        Main folder. '.ac' file and global average needed.
    Orien_para : (2-element-turple)
        (Runname,method) of orientation run. Set (False,False) to skip.
    OD_para : (2-element-turple)
        (Runname,method) of OD run. Set (False,False) to skip.
    Hue_para : (3-element-turple)
        (Runname,method,Hue Para) of Hue run. Set (False,False,False) to skip.
    used_frame : (list), optional
        Used frame in each single condition. The default is [4,5](For GA 1.3Hz).

    Returns
    -------
    Tuning_Property_Dic : (Dic)
        Dictionary of cell tuning property.

    '''
    print('Cell Data need to be processed first.')
    Cp = Cell_Processor(day_folder)
    acn = Cp.all_cell_names
    # Initialize.
    Tuning_Property_Dic = {}
    for i in range(len(acn)):
        Tuning_Property_Dic[acn[i]] = {}
        Tuning_Property_Dic[acn[i]]['_Significant_Tunings'] = []
        
    #%% Calculate Orientation tunings
    if Orien_para[0] != False:
        Orien_Tuning_Dic = Tuning_IDs(Orien_para[1]+'_Orien')
        all_oriens = list(Orien_Tuning_Dic.keys())
        for i in range(len(all_oriens)):
            c_orien = all_oriens[i]
            c_tuning_ids =Orien_Tuning_Dic[c_orien]
            c_orien_tunings = Cp.Index_Calculator_Core(Orien_para[0],c_tuning_ids[0],c_tuning_ids[1],thres = sig_thres)
            for j,ccn in enumerate(acn):
                if c_orien_tunings[acn[j]] != None:
                    ccn = acn[j]
                    Tuning_Property_Dic[ccn][c_orien] = c_orien_tunings[ccn]
                    if c_orien_tunings[ccn]['Significant'] and c_orien_tunings[ccn]['t_value']>0:
                        Tuning_Property_Dic[ccn]['_Significant_Tunings'].append(c_orien)
    else:
        print('No Orientation Run.')
        
    #%% Get best tuning & best G8 tuning(for OD)
    for i,ccn in enumerate(acn):
        cc = Tuning_Property_Dic[ccn]
        c_sig_tuning = cc['_Significant_Tunings']
        c_sig_tuning_G8 = list(set(c_sig_tuning) & set(['Orien0','Orien45','Orien90','Orien135']))
        if c_sig_tuning == []: # if no tuning
            Tuning_Property_Dic[ccn]['_Best_Orien'] = 'No_Tuning'
        else:
            best_tuning = c_sig_tuning[0]
            for i,ct in enumerate(c_sig_tuning):
                if cc[best_tuning]['Cohen_D']<cc[ct]['Cohen_D']:
                    best_tuning = ct
            Tuning_Property_Dic[ccn]['_Best_Orien'] = best_tuning
        if c_sig_tuning_G8 == []:# if no G8 tuning
            Tuning_Property_Dic[ccn]['_Best_Orien_G8'] = 'No_Tuning'
        else:
            best_tuning = c_sig_tuning_G8[0]
            for i,ct in enumerate(c_sig_tuning_G8):
                if cc[best_tuning]['Cohen_D']<cc[ct]['Cohen_D']:
                    best_tuning = ct
            Tuning_Property_Dic[ccn]['_Best_Orien_G8'] = best_tuning
    #%% Get OD tuning from best orientation.
    if OD_para[0] == False:
        print('No OD runs.')
    else:
        OD_Tuning_Dic = Tuning_IDs(OD_para[1])
        all_ods = list(OD_Tuning_Dic.keys())
        for i in range(len(all_ods)):
            c_od = all_ods[i]
            c_tuning_ids =OD_Tuning_Dic[c_od]
            c_od_tunings = Cp.Index_Calculator_Core(OD_para[0],c_tuning_ids[0],c_tuning_ids[1],thres = sig_thres)
            for j in range(len(acn)):
                ccn = acn[j]
                if c_od_tunings[acn[j]] != None:
                    Tuning_Property_Dic[ccn][c_od] = c_od_tunings[ccn]
                    if c_od_tunings[ccn]['Significant'] and c_od_tunings[ccn]['t_value']>0:
                        Tuning_Property_Dic[ccn]['_Significant_Tunings'].append(c_od)
    #%% Get OD Tuning Property
    for i,ccn in enumerate(acn):
        cc = Tuning_Property_Dic[ccn]
        if 'LE' in cc['_Significant_Tunings']:
            Tuning_Property_Dic[ccn]['_OD_Preference'] = 'LE'
        elif 'RE' in cc['_Significant_Tunings']:
            Tuning_Property_Dic[ccn]['_OD_Preference'] = 'RE'
        else:
            Tuning_Property_Dic[ccn]['_OD_Preference'] = 'No_Tuning'
        

    #%% Calculate Direction Tunings.
    if Orien_para[1] == 'G8_2P' or Orien_para[1] == 'G16_2P':
        Dir_Tuning_Dic = Tuning_IDs(Orien_para[1]+'_Dir')
        all_dirs = list(Dir_Tuning_Dic.keys())
        for i,c_dir in enumerate(all_dirs):
            c_tuning_ids = Dir_Tuning_Dic[c_dir]
            c_dir_tunings = Cp.Index_Calculator_Core(Orien_para[0],c_tuning_ids[0],c_tuning_ids[1],thres = sig_thres)
            for j,ccn in enumerate(acn):
                if c_dir_tunings[ccn] != None:
                    Tuning_Property_Dic[ccn][c_dir] = c_dir_tunings[ccn]
                    if c_dir_tunings[ccn]['Significant'] and c_dir_tunings[ccn]['t_value']>0:
                        Tuning_Property_Dic[ccn]['_Significant_Tunings'].append(c_dir)
    else:
        print('No_Direction_Run.')
    #%% Calculate Best Direction Tunings.
    for i,ccn in enumerate(acn):
        all_tunings = Tuning_Property_Dic[ccn]['_Significant_Tunings']
        all_dir_tunings = []
        # Get all direction tunings
        for j,ct in enumerate(all_tunings):
            if 'Dir' in ct:
                all_dir_tunings.append(ct)
        # Get best direction tuning.
        if all_dir_tunings == []:
            Tuning_Property_Dic[ccn]['_Best_Dir'] = 'No_Tuning'
        else:
            best_dir = all_dir_tunings[0]
            for j,ct in enumerate(all_dir_tunings):
                if Tuning_Property_Dic[ccn][ct]['Cohen_D']>Tuning_Property_Dic[ccn][best_dir]['Cohen_D']:
                    best_dir = ct
            Tuning_Property_Dic[ccn]['_Best_Dir'] = best_dir
#%% Calculate color Tunings.
    if Hue_para[0] == False:
        print('No Color Run.')
    else:
        Hue_Tuing_Dic = Tuning_IDs(Hue_para[1],Hue_para[2])
        all_hues = list(Hue_Tuing_Dic.keys())
        for i,c_hue in enumerate(all_hues):
            c_tuning_ids = Hue_Tuing_Dic[c_hue]
            c_hue_tuning = Cp.Index_Calculator_Core(Hue_para[0],c_tuning_ids[0],c_tuning_ids[1],thres = sig_thres)
            for j,ccn in enumerate(acn):
                if c_hue_tuning != None:
                    Tuning_Property_Dic[ccn][c_hue] = c_hue_tuning[ccn]
                    if c_hue_tuning[ccn]['Significant'] and c_hue_tuning[ccn]['t_value']>0:
                        Tuning_Property_Dic[ccn]['_Significant_Tunings'].append(c_hue)
                        
#%% Last, save tuning property and get tuning check lists.
    ot.Save_Variable(day_folder, 'All_Tuning_Property', Tuning_Property_Dic,'.tuning')
    tuning_checklist = Get_Tuning_Checklists(day_folder)
    return Tuning_Property_Dic,tuning_checklist

