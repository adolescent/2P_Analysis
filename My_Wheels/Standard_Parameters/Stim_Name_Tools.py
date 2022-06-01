# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 14:28:17 2021

@author: ZR
"""
import numpy as np
#%% Function1, return ID cobine for response curve.
def Stim_ID_Combiner(mode,para_dic = None):
    
    Stim_IDs = {}
    if mode == 'G8_2P_Oriens':
        print('Orientation 0 is horizontal, counterclockwise.')
        Stim_IDs['Orien0.0'] = [1,5]
        Stim_IDs['Orien45.0'] = [2,6]
        Stim_IDs['Orien90.0'] = [3,7]
        Stim_IDs['Orien135.0'] = [4,8]
        Stim_IDs['Blank'] = [0]
    
    if mode == 'G16_Oriens':
        print('Orientation 0 is horizontal, counterclockwise.')
        Stim_IDs['Orien0.0'] = [1,9]
        Stim_IDs['Orien22.5'] = [2,10]
        Stim_IDs['Orien45.0'] = [3,11]
        Stim_IDs['Orien67.5'] = [4,12]
        Stim_IDs['Orien90.0'] = [5,13]
        Stim_IDs['Orien112.5'] = [6,14]
        Stim_IDs['Orien135.0'] = [7,15]
        Stim_IDs['Orien157.5'] = [8,16]
        Stim_IDs['Blank'] = [0]
    
    if mode == 'OD_2P_Oriens':
        print('Orientation 0 is horizontal, counterclockwise.')
        Stim_IDs['Orien0.0'] = [1,2]
        Stim_IDs['Orien45.0'] = [3,4]
        Stim_IDs['Orien90.0'] = [5,6]
        Stim_IDs['Orien135.0'] = [7,8]
        Stim_IDs['Blank'] = [0]
        
    elif mode == 'G16_Dirs':
        print('Dir 0 is horizontal, moving up. Counterclockwise.')
        print('Subshape recommend:(3,8)')
        for i in range(16):
            current_name = 'Dir'+str(i*22.5)
            Stim_IDs[current_name] = [i+1]
        Stim_IDs['Blank'] = [0]
        Stim_IDs['All'] = list(range(1,17))
        
    elif mode == 'G16_Radar':
        print('Dir 0 is horizontal, moving up. Counterclockwise.')
        for i in range(16):
            current_name = 'Dir'+str(i*22.5)
            Stim_IDs[current_name] = [i+1]
        
    elif mode == 'Color7Dir8_Colors':
        Stim_IDs['Red'] = list(range(1,9))
        Stim_IDs['Yellow'] = list(range(9,17))
        Stim_IDs['Green'] = list(range(17,25))
        Stim_IDs['Cyan'] = list(range(25,33))
        Stim_IDs['Blue'] = list(range(33,41))
        Stim_IDs['Purple'] = list(range(41,49))
        Stim_IDs['White'] = list(range(49,57))
        Stim_IDs['All'] = list(range(1,57))
        
    elif mode == 'Hue7Orien4_Colors':
        Stim_IDs['Red'] = [1,8,15,22]
        Stim_IDs['Yellow'] = [2,9,16,23]
        Stim_IDs['Green'] = [3,10,17,24]
        Stim_IDs['Cyan'] = [4,11,18,25]
        Stim_IDs['Blue'] = [5,12,19,26]
        Stim_IDs['Purple'] = [6,13,20,27]
        Stim_IDs['White'] = [7,14,21,28]
        Stim_IDs['All'] = list(range(1,29))
        
        
    elif mode == 'OD_2P':
        print('1357L,2468R,12 up moving.')
        print('Subshape recommend:(3,5)')
        Stim_IDs['L_All'] = [1,3,5,7]
        Stim_IDs['L_Orien0'] = [1]
        Stim_IDs['L_Orien45'] = [3]
        Stim_IDs['L_Orien90'] = [5]
        Stim_IDs['L_Orien135'] = [7]
        Stim_IDs['R_All'] = [2,4,6,8]
        Stim_IDs['R_Orien0'] = [2]
        Stim_IDs['R_Orien45'] = [4]
        Stim_IDs['R_Orien90'] = [6]
        Stim_IDs['R_Orien135'] = [8]
        Stim_IDs['All'] = list(range(1,9))
        Stim_IDs['Blank'] = [0]
        
    elif mode == 'OD_2P_Radar':
        print('Symetric radar id provided, angle baise = 22.5 advised.')
        Stim_IDs['L_deg0'] = [1]
        Stim_IDs['L_deg45'] = [3]
        Stim_IDs['L_deg90'] = [5]
        Stim_IDs['L_deg135'] = [7]
        Stim_IDs['R_deg135'] = [8]
        Stim_IDs['R_deg90'] = [6]
        Stim_IDs['R_deg45'] = [4]
        Stim_IDs['R_deg0'] = [2]
        
    elif mode == 'Shape3Dir8_General':
        print('General properties of S3D8.ID 1 moving right. Draw in 3*4')
        Stim_IDs['Bars'] =list(range(1,9))
        Stim_IDs['Triangles'] = list(range(9,17))
        Stim_IDs['Circles'] = list(range(17,25))
        Stim_IDs['All'] = list(range(1,25))
        Stim_IDs['Dir0'] = [3,11,19]
        Stim_IDs['Dir45'] = [4,12,20]
        Stim_IDs['Dir90'] = [5,13,21]
        Stim_IDs['Dir135'] = [6,14,22]
        Stim_IDs['Dir180'] = [7,15,23]
        Stim_IDs['Dir225'] = [8,16,24]
        Stim_IDs['Dir270'] = [1,9,17]
        Stim_IDs['Dir315'] = [2,10,18]
        
        
        
        
    elif mode == 'Shape3Dir8_Single':
        print('Generate single condition S3D8 subplots.ID 1 moving right. Draw in 4*8')
        for i in range(8):
            c_name = 'All_Dir'+str(i*45)
            if i >5:
                Stim_IDs[c_name] = [i-5,i+3,i+11]
            else:
                Stim_IDs[c_name] = [i+3,i+11,i+19]
                
        for i in range(8):
            c_name = 'Bar_Dir'+str(i*45)
            if i >5:
                Stim_IDs[c_name] = [i-5]
            else:
                Stim_IDs[c_name] = [i+3]
        for i in range(8):
            c_name = 'Triangle_Dir'+str(i*45)
            if i >5:
                Stim_IDs[c_name] = [i+3]
            else:
                Stim_IDs[c_name] = [i+11]
        for i in range(8):
            c_name = 'Circle_Dir'+str(i*45)
            if i >5:
                Stim_IDs[c_name] = [i+11]
            else:
                Stim_IDs[c_name] = [i+19]
                
    elif mode == 'RFSize': # RF Size & Dir tunings
        if para_dic == None:
            raise IOError('Please give RF Size dics.')
        sizes = para_dic['Size']
        dirs = para_dic['Dir']
        cond_num = len(sizes)*len(dirs)
        size_num = len(sizes)
        dir_num = len(dirs)
        Stim_IDs['All'] = list(range(1,cond_num+1))
        # Then all sizes
        for i in range(size_num):
            c_size = 'Size_'+str(sizes[i])
            Stim_IDs[c_size] = list(range(i*dir_num+1,(i+1)*dir_num+1))
        # Then all dirs
        for i in range(dir_num):
            c_dir = 'Dir'+str(dirs[i])
            Stim_IDs[c_dir] = list(np.arange(i+1,40,dir_num))
            
    elif mode == 'RFSize_SC': # Single Condition RF Size graphs.
        if para_dic == None:
            raise IOError('Please give RF Size dics.')
        sizes = para_dic['Size']
        dirs = para_dic['Dir']
        cond_num = len(sizes)*len(dirs)
        size_num = len(sizes)
        dir_num = len(dirs)
        # get all size average
        for i in range(dir_num):
            Stim_IDs['All_Dir'+str(dirs[i])] = list(np.arange(i+1,41,dir_num))
        
        for i in range(size_num):
            c_size = 'Size_'+str(sizes[i])
            for j in range(dir_num):
                c_dir = c_dir = '_Dir'+str(dirs[j])
                Stim_IDs[c_size+c_dir] = [i*dir_num+j+1]
                
    elif mode == 'HueNOrien4_Color':
        if para_dic == None:
            raise IOError('Please give hue dic.')
        all_hue = para_dic['Hue']
        hue_num = len(all_hue)
        cond_num = hue_num*4+1
        Stim_IDs['All'] = list(range(1,cond_num))
        Stim_IDs['Blank'] = [0]
        for i in range(hue_num):
            c_hue = all_hue[i]
            Stim_IDs[c_hue] = [1+i,hue_num*1+i+1,hue_num*2+i+1,hue_num*3+i+1]
        Stim_IDs['Orien0'] = list(range(1,hue_num+1))
        Stim_IDs['Orien45'] = list(range(hue_num+1,hue_num*2+1))
        Stim_IDs['Orien90'] = list(range(hue_num*2+1,hue_num*3+1))
        Stim_IDs['Orien135'] = list(range(hue_num*3+1,hue_num*4+1))
    elif mode == 'HueNOrien4_SC':
        print('Generate Single condition Hue response dic.')
        print('Subshape recommend:(6,N)')
        all_hue = para_dic['Hue']
        hue_num = len(all_hue)
        orien_lists = [0,45,90,135]
        for i in range(4):
            c_orien = orien_lists[i]
            for j in range(hue_num):
                c_name = all_hue[j]+'_Orien'+str(c_orien)
                Stim_IDs[c_name] = [i*hue_num+j+1]
        for i in range(hue_num):
            Stim_IDs[all_hue[i]+'_All'] = [i+1,i+1+hue_num,i+1+hue_num*2,i+1+hue_num*3]
        Stim_IDs['All'] = list(range(1,hue_num*4+1))
        Stim_IDs['Blank'] = [0]
        
        
    return Stim_IDs
#%% Function 2, get IDs for tuning calculation.
def Tuning_IDs(mode,para_dic = None):
    Tuning_Dics = {}
    if mode == 'G8_2P_Orien':
        print('ID 1 is moving up.')
        for i in range(4):
            c_orien_name = 'Orien'+str(i*45)
            Tuning_Dics[c_orien_name] = [[i%8+1,(i+4)%8+1],[(i+2)%8+1,(i+6)%8+1]]
    
    elif mode =='G8_2P_Dir':
        print('ID 1 is moving up.')
        for i in range(8):
            c_orien_name = 'Dir'+str(i*45)
            Tuning_Dics[c_orien_name] = [[i%8+1],[(i+4)%8+1]]
    
    elif mode == 'G16_2P_Orien':
        print('ID 1 is moving up.')
        for i in range(8):
            if i%2 ==0:
                c_orien_name = 'Orien'+str(int(i*22.5))
            else:
                c_orien_name = 'Orien'+str(i*22.5)
            Tuning_Dics[c_orien_name] = [[i%16+1,(i+8)%16+1],[(i+4)%16+1,(i+12)%16+1]]
    
    elif mode == 'G16_2P_Dir':
        print('ID 1 is moving up.')
        for i in range(8):
            if i%2 == 0:
                c_dir_name = 'Dir'+str(int(i*22.5))
            else:
                c_dir_name = 'Dir'+str(i*22.5)
            Tuning_Dics[c_dir_name] = [[i%16+1],[(i+8)%16+1]]
            
    elif mode == 'OD_2P_Orien':
        print('Use OD for orientation, ID 1 is moving up.')
        Tuning_Dics['Orien0'] = [[1,2],[5,6]]
        Tuning_Dics['Orien45'] = [[3,4],[7,8]]
        Tuning_Dics['Orien90'] = [[5,6],[1,2]]
        Tuning_Dics['Orien135'] = [[7,8],[3,4]]
        
    elif mode == 'OD_2P':
        Tuning_Dics['LE'] = [[1,3,5,7],[2,4,6,8]]
        Tuning_Dics['RE'] = [[2,4,6,8],[1,3,5,7]]
        
    elif mode == 'RGLum':
        Tuning_Dics['RG'] = [[1,2],[3,4]]
        Tuning_Dics['Lum'] = [[3,4],[1,2]]
        
    elif mode == 'HueNOrien4':
        all_hue = para_dic['Hue']
        hue_num = len(all_hue)
        white_id = all_hue.index('White')
        white_conds = [white_id+1,white_id+1+hue_num,white_id+1+hue_num*2,white_id+1+hue_num*3]
        for i,c_hue in enumerate(all_hue):
            if i != white_id:
                Tuning_Dics[c_hue] = [[i+1,i+1+hue_num,i+1+hue_num*2,i+1+hue_num*3],white_conds]
        
    return Tuning_Dics