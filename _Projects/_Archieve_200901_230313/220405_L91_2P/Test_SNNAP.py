# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 15:00:04 2022

@author: ZR
"""

import glob
import tifffile
import OS_Tools_Kit as ot
import cv2
img_frames = ot.Get_File_Name(r'G:\Test_Data\2P\220326_L76\1-004\Results\Final_Aligned_Frames')


# =============================================================================
# with tifffile.TiffWriter(r'G:\Test_Data\2P\220326_L76\1-004\Stack.tif',) as stack: 
#     for filename in img_frames: 
#         stack.save(tifffile.imread(filename))
#         
# =============================================================================
        
        
from skimage.external import tifffile as tif
import numpy as np

image = np.zeros((939,512,512), dtype=np.uint16)
for i in range(len(img_frames)):
    c_graph = cv2.imread(img_frames[i],-1)
    image[i,:,:] = c_graph
    
tif.imsave(r'G:\Test_Data\2P\220326_L76\_SNNAP_Test\test.tif', image)
