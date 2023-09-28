# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 12:40:11 2019

@author: ZR

Head & Frame Reader for VDAQ blk files.
BLk files captured by VDAQ have the formation of uint32, and have a head file of 429 int units(1716 bites)

"""
#%%
import numpy as np
import struct

class BLK_Reader(object):
    
    name = r'Methood used for blk reading..'
    
    def BLK_Head_Read(self,file_path):#读头文件，注意对VDAQ系统而言，以下的编码方式和位置都是固定格式，前1716字节就是这么分配的，No Reason = =。
        self.BLK_Property = {}#定义空字典，作为储存的空间
        temp_data = np.fromfile(file_path,dtype = 'u4')[0:429]#前429为头文件
        self.BLK_Property['Data_type'] = temp_data[7]#存储数据类型，12为uint16；13为uint32；14为float32.一般都是13
        self.BLK_Property['Width'] = temp_data[9]
        self.BLK_Property['Height'] = temp_data[10]
        self.BLK_Property['N_Frame_Per_Stim'] = temp_data[11]
        self.BLK_Property['N_Stim'] = temp_data[12]
        del temp_data
        #接下来更换编码模式，用字符模式读取文件信息。
        binFile=open(file_path,'rb')
        binFile.seek(100)
        current_read = binFile.read(16)
        self.BLK_Property['Date'] = struct.unpack('16c',current_read)
        binFile.seek(672)
        current_read = binFile.read(256)
        self.BLK_Property['Stim_ID_Sequence'] = struct.unpack('256c',current_read)
        binFile.close()
        
    def Data_Acquisition(self,BLK_Property,blk_path):
        
        Height = BLK_Property['Height']
        Width = BLK_Property['Width']
        N_Stim = BLK_Property['N_Stim']
        Frame_Per_Stim = BLK_Property['N_Frame_Per_Stim']
        
        data = np.fromfile(blk_path, dtype='<u4')[429:]
        all_graph = np.reshape(data,(-1,Height,Width))
        #检查可行性，确认没有bug了再继续
        if np.shape(all_graph)[0] != N_Stim*Frame_Per_Stim:
            raise Exception('Frame Size not Match, Unexpected error.')
            
        self.BLK_Frames = {}        
        for i in range(N_Stim):
            start = i*Frame_Per_Stim
            stop = (i+1)*Frame_Per_Stim
            self.BLK_Frames[i+1] = all_graph[start:stop,:,:].astype('f8')
        
        
    def Direct_BLK_Read(self,file_path):
        
        self.BLK_Head_Read(file_path)
        self.Data_Acquisition(self.BLK_Property,file_path)
        
        return(self.BLK_Frames)
        
        
    
#%%
if __name__ == '__main__':
    file_path = r'E:\ZR\Data_Temp\180629_L63_OI_Run01_G8_Test\G8_E00B000.BLK'
    BR = BLK_Reader()
    test = BR.Direct_BLK_Read(file_path)