# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 15:54:03 2019

@author: ZR
"""

#%%Filter,可考虑情况不同加新的filter内容进去。注意输出的文件格式，和输入的格式往往相同，减法要小心。
import numpy as np
import scipy.ndimage

class Filter(object):
    
    name = r'Easy filter for data'
    def Gaussian_Filter(self,Input_Graph,Mask_shape,std):
        #先计算得到高斯掩膜
        m,n = [(ss-1.)/2. for ss in Mask_shape] #得到横纵方向的半高宽
        y,x = np.ogrid[-m:m+1,-n:n+1]    #左闭右开，得到每一个取值。
        mask = np.exp( -(x*x + y*y) / (2.*std*std) )
        mask[ mask < np.finfo(mask.dtype).eps*mask.max() ] = 0 #小于精度的置为零。
        sumh = mask.sum() # 归一化高斯掩模，使整个模的和为1
        if sumh != 0:
            mask /= sumh
        #以上得到了高斯掩膜，之后做互相关并返回之。
        Filtered_graph = scipy.ndimage.correlate(Input_Graph,mask,mode = 'nearest')
        return Filtered_graph
        
        
    def Main(self,Input_Graph,method_key,parameters):
        
        if method_key == 'Gaussian':#对高斯滤波
            Mask_shape = parameters[0]
            std = parameters[1]
            self.Filtered_data = self.Gaussian_Filter(Input_Graph,Mask_shape,std)
            return self.Filtered_data
        else:
            raise Exception('Filter Method not supported yet, please contace me.')
            
            
if __name__ == '__main__':
    F = Filter()
    b = F.Main(a,'Gaussian',[[2,2],2])