'''
This function helps you to read the analog bin file in data folder.


'''
#%%
import OS_Tools_Kit as ot
import struct
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt


def Analog_Bin_Reader(path):

    # open file in read-bin mode
    with open(path, mode='rb') as file: # b is important -> binary
        header_bytes = file.read(20) # in Ois system, the first 5 4bit are header info
        data_bytes = file.read()

    # Headers are in format:[version,N_channels,Capture_Rate,Max_Capture_Rate,Max_Capture_Seconds] 
    header = struct.unpack('5i', header_bytes) # headers are int format(i)
    print(f"Number of analog channels : {header[1]}")
    data = struct.unpack(f'{len(data_bytes)//8}d', data_bytes) # data are double format(d)
    

    # save data in shape(N_capture*N_channle)
    data_matrix = np.reshape(np.array(data),(-1,header[1]))
    # and transform header in np array.
    header_info = np.array(header)


    return header_info,data_matrix







#%% Test run part 

if __name__ == '__main__':
    binfile_path = r'D:\#FDU\240306_New_Imaging_Setup_Tests\ai_00000.bin'
    header,data = Analog_Bin_Reader(binfile_path)