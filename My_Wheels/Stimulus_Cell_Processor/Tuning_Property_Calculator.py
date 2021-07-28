# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 14:14:49 2021

@author: ZR
"""

from Cell_Processor import Cell_Processor
from Standard_Parameters.Stim_Name_Tools import Tuning_IDs
def Tuning_Property_Calculator(day_folder,
                               Orien_para = ('Run002','G16_2P'),
                               OD_para = ('Run006','OD_2P'),
                               Hue_para = ('Run007','HueNOrien4',{'Hue':['Red','Yellow','Green','Cyan','Blue','Purple','White']}),
                               used_frame = [4,5]
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
    # First, calculate Orien & Dir tuning.
    Orien_Tuning_Dic = Tuning_IDs(Orien_para+'_Orien')

    return Tuning_Property_Dic