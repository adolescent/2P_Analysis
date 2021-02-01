# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 13:18:31 2020

@author: ZR
"""
import numpy as np

def Sub_Dic_Generator(mode,para_dic = None):
    '''
    Generate subtraction dics.

    Parameters
    ----------
    mode : ('G8','G8+90','RGLum4','OD_OI','OD_2P')
        Mode of data. This is important for 
    para_dic : (Dic)
        parameters may need for some run such as SF3 TF3 or etc...
    


    Returns
    -------
    sub_dics : TYPE
        DESCRIPTION.

    '''
    sub_dics = {}
    if mode == 'OD_OI':
        sub_dics['All-0'] = [[1,2,3,4,5,6,7,8],[0]]
        sub_dics['OD'] = [[1,2,3,4],[5,6,7,8]]
        sub_dics['L-0'] = [[1,2,3,4],[0]]
        sub_dics['R-0'] = [[5,6,7,8],[0]]
        sub_dics['A-O'] = [[4,8],[2,6]]
        sub_dics['H-V'] = [[3,7],[1,5]]
        sub_dics['HV-AO'] = [[1,3,5,7],[2,4,6,8]]
        
    elif mode == 'OD_2P':
        sub_dics['All-0'] = [[1,2,3,4,5,6,7,8],[0]]
        sub_dics['OD'] = [[1,3,5,7],[2,4,6,8]]
        sub_dics['L-0'] = [[1,3,5,7],[0]]
        sub_dics['R-0'] = [[2,4,6,8],[0]]
        sub_dics['H-V'] = [[1,2],[5,6]]
        sub_dics['A-O'] = [[3,4],[7,8]]
        sub_dics['HV-AO'] = [[1,2,5,6],[3,4,7,8]]
        
    elif mode == 'G8_Norm':# Normal G8, ID1 is right moving bars.
        sub_dics['All-0'] = [[1,2,3,4,5,6,7,8],[0]]
        sub_dics['H-V'] = [[3,7],[1,5]]
        sub_dics['A-O'] = [[4,8],[2,6]]
        sub_dics['HV-AO'] = [[1,3,5,7],[2,4,6,8]]
        sub_dics['Orien0-0'] = [[3,7],[0]]
        sub_dics['Orien45-0'] = [[4,8],[0]]
        sub_dics['Orien90-0'] = [[1,5],[0]]
        sub_dics['Orien135-0'] = [[2,6],[0]]
        sub_dics['DirU-D'] = [[2,3,4],[6,7,8]]
        sub_dics['DirL-R'] = [[4,5,6],[8,1,2]]
        
    elif mode == 'G8+90': # In early 2p stims, use up moving bars as ID1.
        sub_dics['All-0'] = [[1,2,3,4,5,6,7,8],[0]]
        sub_dics['H-V'] = [[1,5],[3,7]]
        sub_dics['A-O'] = [[2,6],[4,8]]
        sub_dics['HV-AO'] = [[1,3,5,7],[2,4,6,8]]
        sub_dics['Orien0-0'] = [[1,5],[0]]
        sub_dics['Orien45-0'] = [[2,6],[0]]
        sub_dics['Orien90-0'] = [[3,7],[0]]
        sub_dics['Orien135-0'] = [[4,8],[0]]
        sub_dics['DirU-D'] = [[2,1,8],[4,5,6]]
        sub_dics['DirL-R'] = [[2,3,4],[6,7,8]]
        
        
    elif mode == 'RGLum4':
        sub_dics['All-0'] = [[1,2,3,4],[0]]
        sub_dics['RG-Lum'] = [[1,2],[3,4]]
        sub_dics['A-O'] = [[1,3],[2,4]]
        sub_dics['RG-0'] = [[1,2],[0]]
        sub_dics['Lum-0'] = [[3,4],[0]]
        
        
    elif mode == 'Color7Dir8':
        print('Use ISI as 0 in this dic.')
        sub_dics['H-V'] = [[3,11,19,27,35,43,51,7,15,23,31,39,47,55],[1,9,17,25,33,41,49,5,13,21,29,37,45,53]]
        sub_dics['A-O'] = [[4,12,20,28,36,44,52,8,16,24,32,40,48,56],[2,10,18,26,34,42,50,6,14,22,30,38,46,54]]
        sub_dics['DirU-D'] = [[2,10,18,26,34,42,50,3,11,19,27,35,43,51,4,12,20,28,36,44,52],[6,7,8,14,15,16,22,23,24,30,31,32,38,39,40,46,47,48,54,55,56]]
        sub_dics['DirL-R'] = [[4,5,6,12,13,14,20,21,22,28,29,30,36,37,38,44,45,46,52,53,54],[2,1,8,10,9,16,18,17,24,26,27,32,34,35,40,42,43,48,50,51,56]]
        sub_dics['Orien0-0'] = [[3,11,19,27,35,43,51,7,15,23,31,39,47,55],[-1]]
        sub_dics['Orien45-0'] = [[4,12,20,28,36,44,52,8,16,24,32,40,48,56],[-1]]
        sub_dics['Orien90-0'] = [[1,9,17,25,33,41,49,5,13,21,29,37,45,53],[-1]]
        sub_dics['Orien135-0'] = [[2,6],[-1]]
        sub_dics['Red-White'] = [[1,2,3,4,5,6,7,8],[49,50,51,52,53,54,55,56]]
        sub_dics['Yellow-White'] = [[9,10,11,12,13,14,15,16],[49,50,51,52,53,54,55,56]]
        sub_dics['Green-White'] = [[17,18,19,20,21,22,23,24],[49,50,51,52,53,54,55,56]]
        sub_dics['Cyan-White'] = [[25,26,27,28,29,30,31,32],[49,50,51,52,53,54,55,56]]
        sub_dics['Blue-White'] = [[33,34,35,36,37,38,39,40],[49,50,51,52,53,54,55,56]]
        sub_dics['Purple-White'] = [[41,42,43,44,45,46,47,48],[49,50,51,52,53,54,55,56]]
        sub_dics['Red-Cyan'] = [[1,2,3,4,5,6,7,8],[25,26,27,28,29,30,31,32]]
        sub_dics['Yellow-Blue'] = [[9,10,11,12,13,14,15,16],[33,34,35,36,37,38,39,40]]
        sub_dics['Green-Purple'] = [[17,18,19,20,21,22,23,24],[41,42,43,44,45,46,47,48]]
    
    
    elif mode == 'Shape3Dir8':
        sub_dics['H-V'] = [[3,7],[1,5]]
        sub_dics['Triangle-Bar'] = [[9,10,11,12,13,14,15,16],[1,2,3,4,5,6,7,8]]
        sub_dics['Circle-Bar'] = [[17,18,19,20,21,22,23,24],[1,2,3,4,5,6,7,8]]
        sub_dics['Circle-Triangle'] = [[17,18,19,20,21,22,23,24],[9,10,11,12,13,14,15,16]]
        sub_dics['DirU-D_All'] = [[2,3,4,10,11,12,18,19,20],[6,7,8,14,15,16,22,23,24]]
        sub_dics['DirL-R_All'] = [[4,5,6,12,13,14,20,21,22],[2,1,8,10,9,16,18,17,24]]
        sub_dics['DirU-D_Bar'] = [[2,3,4],[6,7,8]]
        sub_dics['DirU-D_Circle'] = [[18,19,20],[22,23,24]]
        sub_dics['DirU-D_Triangle'] = [[10,11,12],[14,15,16]]
        sub_dics['DirL-R_Bar'] = [[4,5,6],[2,1,8]]
        sub_dics['DirL-R_Circle'] = [[20,21,22],[24,17,18]]
        sub_dics['DirL-R_Triangle'] = [[12,13,14],[9,10,16]]
        sub_dics['Circle-0'] = [[17,18,19,20,21,22,23,24],[-1]]
        sub_dics['Bars-0'] = [[1,2,3,4,5,6,7,8],[-1]]
        sub_dics['Triangle-0'] = [[9,10,11,12,13,14,15,16],[-1]]
        sub_dics['All-0'] = [list(np.arange(1,25)),[-1]]
        
    elif mode == 'G16_2P':# id 1,Up moving. Counter clockwise.
        sub_dics['All-0'] = [list(np.arange(1,17)),[0]]
        sub_dics['H-V'] = [[1,9],[5,13]]
        sub_dics['A-O'] = [[3,11],[7,15]]
        
        
        
    elif mode == 'SF_TF_Dir':
        if para_dic == None:
            raise IOError('Please give parameter dictionary.')
        
        
    return sub_dics