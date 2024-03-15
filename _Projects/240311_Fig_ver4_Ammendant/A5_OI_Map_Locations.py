'''
This will cut locations and get specified OI Map Locs.

'''

#%%

import OS_Tools_Kit as ot
import cv2
import Graph_Operation_Kit as gt
import numpy as np
import matplotlib.pyplot as plt


L76_path = r'D:\_OI_Test_Datas\L76_OI_Data'
L76_green = cv2.imread(f'{L76_path}\green0_roi2x2.bmp')
L76_6b_mask = cv2.imread(f'{L76_path}\Location6B_Mask.tif',0)
L76_7a_mask = cv2.imread(f'{L76_path}\Location7A_Mask.tif',0)
L85_6b_mask = cv2.imread(r'D:\_OI_Test_Datas\L85_OI_Data\Loc6B_Mask.tif',0)
wp = r'D:\_Path_For_Figs\240312_Figs_v4_A1\A5_All_V2_Locations'

def Mask_Coords(mask_frame):
    nonzero_indices = np.nonzero(mask_frame)
    min_x = np.min(nonzero_indices[1])  # Minimum x-coordinate
    min_y = np.min(nonzero_indices[0])  # Minimum y-coordinate
    max_x = np.max(nonzero_indices[1])  # Maximum x-coordinate
    max_y = np.max(nonzero_indices[0])  # Maximum y-coordinate
    Left_Upper = (min_x,min_y)
    Right_Lower = (max_x,max_y)
    return Left_Upper,Right_Lower

#%% DO Graph of L76 6B Here.
L76_6b_LU,L76_6b_RL = Mask_Coords(L76_6b_mask)
L76_6b_masked_Locs = cv2.rectangle(L76_green,L76_6b_LU,L76_6b_RL,(255,0,0),1) 
plt.imshow(L76_6b_masked_Locs)
od = cv2.imread(r'D:\_OI_Test_Datas\L76_OI_Data\OD\OD_Ttest.bmp')
od_masked_Locs = cv2.rectangle(od,L76_6b_LU,L76_6b_RL,(0,0,255),1) 
# plt.imshow(od_masked_Locs)
ao = cv2.imread(r'D:\_OI_Test_Datas\L76_OI_Data\G8\A-O_Ttest.bmp')
ao_masked_Locs = cv2.rectangle(ao,L76_6b_LU,L76_6b_RL,(0,0,255),1) 
# plt.imshow(ao_masked_Locs)
hv = cv2.imread(r'D:\_OI_Test_Datas\L76_OI_Data\G8\H-V_Ttest.bmp')
hv_masked_Locs = cv2.rectangle(hv,L76_6b_LU,L76_6b_RL,(0,0,255),1) 
rglum = cv2.imread(r'D:\_OI_Test_Datas\L76_OI_Data\RGLum\RG-0_Ttest.bmp')
rglum_masked_Locs = cv2.rectangle(rglum,L76_6b_LU,L76_6b_RL,(0,0,255),1) 
# plt.imshow(rglum_masked_Locs)
cv2.imwrite(f'{wp}\OD_Masked_6B_L76.png',od_masked_Locs)
cv2.imwrite(f'{wp}\AO_Masked_6B_L76.png',ao_masked_Locs)
cv2.imwrite(f'{wp}\HV_Masked_6B_L76.png',hv_masked_Locs)
cv2.imwrite(f'{wp}\RGLum_Masked_6B_L76.png',rglum_masked_Locs)
#%% DO Graph of L76 7A Here.
L76_6b_LU,L76_6b_RL = Mask_Coords(L76_7a_mask)
L76_6b_masked_Locs = cv2.rectangle(L76_green,L76_6b_LU,L76_6b_RL,(255,0,0),1) 
plt.imshow(L76_6b_masked_Locs)
od = cv2.imread(r'D:\_OI_Test_Datas\L76_OI_Data\OD\OD_Ttest.bmp')
od_masked_Locs = cv2.rectangle(od,L76_6b_LU,L76_6b_RL,(0,0,255),1) 
# plt.imshow(od_masked_Locs)
ao = cv2.imread(r'D:\_OI_Test_Datas\L76_OI_Data\G8\A-O_Ttest.bmp')
ao_masked_Locs = cv2.rectangle(ao,L76_6b_LU,L76_6b_RL,(0,0,255),1) 
hv = cv2.imread(r'D:\_OI_Test_Datas\L76_OI_Data\G8\H-V_Ttest.bmp')
hv_masked_Locs = cv2.rectangle(hv,L76_6b_LU,L76_6b_RL,(0,0,255),1) 
# plt.imshow(ao_masked_Locs)
rglum = cv2.imread(r'D:\_OI_Test_Datas\L76_OI_Data\RGLum\RG-0_Ttest.bmp')
rglum_masked_Locs = cv2.rectangle(rglum,L76_6b_LU,L76_6b_RL,(0,0,255),1) 
# plt.imshow(rglum_masked_Locs)
cv2.imwrite(f'{wp}\OD_Masked_7A_L76.png',od_masked_Locs)
cv2.imwrite(f'{wp}\AO_Masked_7A_L76.png',ao_masked_Locs)
cv2.imwrite(f'{wp}\HV_Masked_7A_L76.png',hv_masked_Locs)
cv2.imwrite(f'{wp}\RGLum_Masked_7A_L76.png',rglum_masked_Locs)
#%% DO Graph of L85 6B Here.
L76_6b_LU,L76_6b_RL = Mask_Coords(L85_6b_mask)
# L76_6b_masked_Locs = cv2.rectangle(L76_green,L76_6b_LU,L76_6b_RL,(255,0,0),1) 
# plt.imshow(L76_6b_masked_Locs)
od = cv2.imread(r'D:\_OI_Test_Datas\L85_OI_Data\Run07_OD\OD_Ttest.bmp')
od_masked_Locs = cv2.rectangle(od,L76_6b_LU,L76_6b_RL,(0,0,255),1) 
# plt.imshow(od_masked_Locs)
ao = cv2.imread(r'D:\_OI_Test_Datas\L85_OI_Data\Run02_G8\A-O_Ttest.bmp')
ao_masked_Locs = cv2.rectangle(ao,L76_6b_LU,L76_6b_RL,(0,0,255),1) 
hv = cv2.imread(r'D:\_OI_Test_Datas\L85_OI_Data\Run02_G8\H-V_Ttest.bmp')
hv_masked_Locs = cv2.rectangle(hv,L76_6b_LU,L76_6b_RL,(0,0,255),1) 
# plt.imshow(ao_masked_Locs)
rglum = cv2.imread(r'D:\_OI_Test_Datas\L85_OI_Data\Run09_RGLum4\RG-0_Ttest.bmp')
rglum_masked_Locs = cv2.rectangle(rglum,L76_6b_LU,L76_6b_RL,(0,0,255),1) 
# plt.imshow(rglum_masked_Locs)
cv2.imwrite(f'{wp}\OD_Masked_6B_L85.png',od_masked_Locs)
cv2.imwrite(f'{wp}\AO_Masked_6B_L85.png',ao_masked_Locs)
cv2.imwrite(f'{wp}\HV_Masked_6B_L85.png',hv_masked_Locs)
cv2.imwrite(f'{wp}\RGLum_Masked_6B_L85.png',rglum_masked_Locs)