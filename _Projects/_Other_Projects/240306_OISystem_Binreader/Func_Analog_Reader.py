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
    data_matrix = np.array(data)
    # The data matrix is saved in an 12*10000 cycle matrix. We need to cut them in cycles.
    cycle_num = len(data_matrix)//120000
    all_ai_signals = np.zeros(shape = (cycle_num*10000,12))
    for i in range(cycle_num):
        c_cuts = data_matrix[i*120000:(i+1)*120000]
        c_ai_signals = np.reshape(c_cuts,(12,-1))
        all_ai_signals[i*10000:(i+1)*10000,:] = c_ai_signals.T


    # and transform header in np array.
    header_info = np.array(header)



    return header_info,all_ai_signals







#%% Test run part 

if __name__ == '__main__':
    # binfile_path = r'E:\T2\ai_00000.bin'
    # header,data = Analog_Bin_Reader(binfile_path)
    # plt.plot(data[:,0])
    b2 = r'C:\Users\admin\Documents\WeChat Files\wxid_w0nanzckkqa941\FileStorage\File\2024-03\ai_00000.bin'
    header2,data2 = Analog_Bin_Reader(b2)
    plt.plot(data2[:,0])
    # plt.plot(data2[:,1])
    # plt.plot(data2[:,0])
