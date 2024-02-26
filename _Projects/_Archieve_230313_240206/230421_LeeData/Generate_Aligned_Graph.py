'''
Align and cell find from Li Ming's 2p data.
40 min awake(maybe sleeping? V4 data)
'''
#%%
import h5py
import numpy as np
from My_Wheels.Caiman_API.One_Key_Caiman import One_Key_Caiman
import My_Wheels.OS_Tools_Kit as ot
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tifffile as tif
from tqdm import tqdm
# plt.switch_backend('webAgg')
filepath = r'D:\ZR\_Data_Temp\2pt_T151425_A2\aa.mat'
day_folder = r'D:\ZR\_Data_Temp\2pt_T151425_A2'
#%%
arrays = {}
f = h5py.File(filepath)
for k, v in f.items():
    arrays[k] = np.array(v)
# np.save(day_folder+'\Graph_Stacks.npy',arrays[list(arrays.keys)[0]][:,:,:,0])
# save will cost a fcking year= = transfer data into u2-tif file.
#%%
data = arrays[list(arrays.keys())[0]][:,142:654,:,0]# cut black boulder
del arrays
cut_size = 2000
graph_num = data.shape[0]
width = data.shape[1]
height = data.shape[2]
save_path = day_folder
last_frame_num = graph_num%cut_size
if last_frame_num>500:# if last frame number <100, concat them into file before.
    subfile_num = np.ceil(graph_num/cut_size).astype('int')
else:
    subfile_num = np.round(graph_num/cut_size).astype('int')
    
for j in tqdm(range(subfile_num)):
    c_filename = '1-001_'+str(j+1)+'.tif'
    whole_c_filename = ot.join(save_path,c_filename)
    if j != (subfile_num-1): #not last graph
        c_tif_struct = np.zeros(shape = (cut_size,width,height),dtype='u2')
    else:
        c_tif_struct = np.zeros(shape = (graph_num-cut_size*(subfile_num-1),width,height),dtype='u2')
    
    for k in range(j*cut_size,(j+1)*cut_size):
        c_graph = data[k,:,:].T
        c_tif_struct[k%cut_size,:,:] = c_graph
    tif.imwrite(whole_c_filename,c_tif_struct)
#%% After generation,caiman will do following work
Okc = One_Key_Caiman(day_folder, [1],align_base = '1-001',boulder = (20,20,20,20),fps = 31,decay=0.35)
Okc.Motion_Corr_All()
#Okc.global_avr = cv2.imread(r'G:\Test_Data\2P\220630_L76_2P\_CAIMAN\Summarize\Global_Average_cai.tif',-1)
Okc.Cell_Find(boulders= Okc.boulder)
# Okc.Series_Generator_Low_Memory()
Okc.Series_Generator_NG()
#%% Check data plots.
all_cell_data = ot.Load_Variable(day_folder,r'\_CAIMAN\All_Series_Dic.pkl')

#%%
plt.switch_backend('webAgg')
plt.imshow(a)
plt.show()
