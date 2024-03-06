'''
Try to read the bin file of stim series in data folder.

'''


#%%
import OS_Tools_Kit as ot
import struct
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

wp = r'D:\#FDU\240306_New_Imaging_Setup_Tests'
bin_path = ot.join(wp,'ai_00000.bin')

#%%########################## 1. BIN READER ##########################

rec_channel = 12

with open(bin_path, mode='rb') as file: # b is important -> binary
    header_bytes = file.read(20)
    header = struct.unpack('5i', header_bytes)
    data_bytes = file.read()
    data = struct.unpack(f'{len(data_bytes)//8}d', data_bytes)
    # Print the header values
    print("Header values:", header)
    # Print the data values
    # print("Data values:", data)
    
n_channel = np.array(header)[1]
data_matrix = np.reshape(np.array(data),(-1,12))

