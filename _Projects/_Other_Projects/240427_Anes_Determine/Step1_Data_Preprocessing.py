'''
Preprocess the data into an pandas frame with dics readable.
Follow the method we use before, but standarize and one key process.

'''



#%% Import 

import OS_Tools_Kit as ot
import neo
import numpy as np
from neo.io import PlexonIO
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from scipy.interpolate import interp1d
import mat73


class One_Key_Anes_Process(object):
    name = r'Process data by one key'
    def __init__(self,wp):
        self.wp = wp
        self.case_name = wp.split('\\')[-1]
        print(f'Case : {self.case_name}')
        self.all_bio_rec = ot.Get_Subfolders(wp,'','Relative')
        print(f'Recorded Bio Paras :{self.all_bio_rec}')
        self.save_path = ot.join(wp,'_Processing')
        ot.mkdir(self.save_path)

    def EEG_Reader(self,EMG_ch = 0,EEG_ch = 2,method = 'plx'):
        '''Real in Plexon .plx File, and save them into python readable format.'''
        # check exist.
        if 'EEG' not in self.all_bio_rec:
            raise FileExistsError('Check whether have EEG Recording, or properly renamed')
        if method == 'plx':
            raw_eeg_path = ot.Get_File_Name(ot.join(self.wp,'EEG'),'.plx',include_sub=True)
            if len(raw_eeg_path) != 1:
                raise LookupError('Number of EEG File not right, check plz.')
        # Load EEG and EMG Data.
            reader = PlexonIO(raw_eeg_path[0])
            blks = reader.read()
            eeg_signals_raw = np.array(blks[0].segments[0].analogsignals[0])
            self.EEG_fps = float(blks[0].segments[0].analogsignals[0].sampling_rate)
            self.EEG_Signal = eeg_signals_raw[:,EEG_ch]
            self.EMG_Signal = eeg_signals_raw[:,EMG_ch]
            
        if method == 'smr':
            raw_eeg_path = ot.Get_File_Name(ot.join(self.wp,'EEG'),'.smr',include_sub=True)
            if len(raw_eeg_path) != 1:
                raise LookupError('Number of EEG File not right, check plz.')
        # Load EEG and EMG Data. Use SMR method here.
            reader = neo.io.Spike2IO(filename=(raw_eeg_path[0]),try_signal_grouping=False)
            blks = reader.read(lazy=False)[0]
            eeg_signals_raw = np.array(blks.segments[0].analogsignals)[:,:,0]
            self.EEG_fps = float(blks.segments[0].analogsignals[0].sampling_rate)
            self.EEG_Signal = eeg_signals_raw[EEG_ch,:]
            self.EMG_Signal = eeg_signals_raw[EMG_ch,:]
            # blks = reader.read()
            # eeg_signals_raw = np.array(blks[0].segments[0].analogsignals[0])

        

        
        np.save(ot.join(self.save_path,'EEG_Raw'),self.EEG_Signal)
        np.save(ot.join(self.save_path,'EMG_Raw'),self.EMG_Signal)

    def FFT_Core(self,input_matrix,freq_bin = 0.5,fps = 1000,ratio = False):
        input_matrix = np.array(input_matrix)
        # get raw frame spectrums.
        all_specs = np.zeros(shape = ((input_matrix.shape[0]// 2)-1,input_matrix.shape[1]),dtype = 'f8')
        for i in range(input_matrix.shape[1]):
            c_series = input_matrix[:,i]
            c_fft = np.fft.fft(c_series)
            power_spectrum = np.abs(c_fft)[1:input_matrix.shape[0]// 2] ** 2
            if ratio == True:
                power_spectrum = power_spectrum/power_spectrum.sum()
            all_specs[:,i] = power_spectrum
        binnum = int(fps/(2*freq_bin))
        binsize = round(len(all_specs)/binnum)
        binned_freq = np.zeros(shape = (binnum,input_matrix.shape[1]),dtype='f8')
        for i in range(binnum):
            c_bin_freqs = all_specs[i*binsize:(i+1)*binsize,:].sum(0)
            binned_freq[i,:] = c_bin_freqs
        return binned_freq
        
    def EEG_Slide_Power(self,winsize = 60,step = 1):
        '''Transfer EEG data into Power data, Paras given in second.'''
        win_pointnum = winsize*self.EEG_fps
        step_pointnum = step*self.EEG_fps
        used_eeg = self.EEG_Signal
        winnum = int(((len(used_eeg)-win_pointnum)//step_pointnum)+1)
        freq_bin = 0.5
        slided_ffts = np.zeros(shape = (int(self.EEG_fps/(2*freq_bin)),winnum),dtype = 'f8')
        for i in tqdm(range(winnum)):
            c_win = used_eeg[int(i*step_pointnum):int(i*step_pointnum+win_pointnum)]
            slided_ffts[:,i] = self.FFT_Core(c_win.reshape(-1,1),freq_bin,self.EEG_fps).flatten()
        # save EEG spectrum power.
        self.all_power = {}
        self.all_power['Sepctrum'] = slided_ffts
        self.all_power['Delta'] = slided_ffts[1:8,:].sum(0) # 0.5-4Hz
        self.all_power['Theta'] = slided_ffts[8:14,:].sum(0) # 4-7Hz
        self.all_power['Alpha'] = slided_ffts[16:26,:].sum(0) # 8-13Hz
        self.all_power['Beta'] = slided_ffts[26:60,:].sum(0) # 13-30Hz
        self.all_power['All_EEG_Power'] = slided_ffts.sum(0)
        self.all_power['theta_delta_ratio'] = self.all_power['Theta']/self.all_power['Delta']
        ot.Save_Variable(self.save_path,'EEG_Power',self.all_power)
        # calcualte and return EEG capture time.
        EEG_time = len(self.all_power['All_EEG_Power'])/step
        return EEG_time
    
    def MouseOX_Reader(self,fps = 15,down_to = 1):

        if 'MouseOX' not in self.all_bio_rec:
            raise FileExistsError('Check whether have Body Temp Recording, or properly renamed')
        raw_ox_path = ot.Get_File_Name(ot.join(self.wp,'MouseOX'),'.csv',include_sub=True)
        if len(raw_ox_path) != 1:
            raise LookupError('Number of EEG File not right, check plz.')
        ox_data = pd.read_csv(raw_ox_path[0],skiprows=3,header=None,encoding='latin-1')
        useful_ox_data = ox_data[[5,6,7,8]]
        useful_ox_data.columns=['SpO2','HR','Resp','Pulse_Distention']

        # reading done, down sample here.
        original_time = np.arange(len(useful_ox_data))/fps
        new_time = np.arange(0,round(original_time[-1]),1/down_to)
        self.OX_datas = pd.DataFrame(0.0,columns = ['SpO2','HR','Resp','Pulse_Distention'],index = new_time)
        for i,c_para in enumerate(['SpO2','HR','Resp','Pulse_Distention']):
            interpolated_spo2 = interp1d(original_time, useful_ox_data[c_para], kind='linear')
            self.OX_datas[c_para] = interpolated_spo2(new_time)

    def Pad_Reader(self,fps = 10,down_to = 1):
        if 'Pad' not in self.all_bio_rec:
            raise FileExistsError('Check whether have Pad Recording, or properly renamed')
        raw_pad_path = ot.Get_File_Name(ot.join(self.wp,'Pad'),'.mat',include_sub=True)
        if len(raw_pad_path) != 1:
            raise LookupError('Number of Pad File not right, check plz.')
        pad_move_data = mat73.loadmat(raw_pad_path[0])['proc']['runSpeed']
        pad_move = np.linalg.norm(pad_move_data,axis=1)
        # reading done, down sample data.
        original_time = np.arange(len(pad_move))/fps
        new_time = np.arange(0,round(original_time[-1]),1/down_to)
        interpolated_pad = interp1d(original_time, pad_move, kind='linear')
        self.Pad_Movement = interpolated_pad(new_time)

    def Pupil_Reader(self,fps = 10,down_to = 1):
        if 'Pupil' not in self.all_bio_rec:
            raise FileExistsError('Check whether have Pupil Recording, or properly renamed')
        raw_pupil_path = ot.Get_File_Name(ot.join(self.wp,'Pupil'),'.mat',include_sub=True)
        if len(raw_pupil_path) != 1:
            raise LookupError('Number of Pupil File not right, check plz.')
        pupil_data = mat73.loadmat(raw_pupil_path[0])['proc']['pupil']['area']
        # reading done, down sample data.
        original_time = np.arange(len(pupil_data))/fps
        new_time = np.arange(0,round(original_time[-1]),1/down_to)
        interpolated_pupil = interp1d(original_time,pupil_data, kind='linear')
        self.Pupil = interpolated_pupil(new_time)

    def Runspeed_Reader(self,down_to = 1):
        if 'RunningSpeed' not in self.all_bio_rec:
            raise FileExistsError('Check whether have Run Speed Recording, or properly renamed')
        raw_runspeed_path = ot.Get_File_Name(ot.join(self.wp,'RunningSpeed'),'.txt',include_sub=True)
        if len(raw_runspeed_path) != 1:
            raise LookupError('Number of Runspeed File not right, check plz.')
        runspeed = pd.read_csv(raw_runspeed_path[0],sep=r'	',skiprows=1,header=None)
        runspeed.columns=['Time','X1','Y1','X2','Y2']
        runspeed_x = np.array(abs(runspeed['X1']-runspeed['X2']))
        runspeed_y = np.array(abs(runspeed['Y1']-runspeed['Y2']))
        all_speed = np.sqrt(runspeed_x**2+runspeed_y**2)

        original_time = np.array(runspeed['Time']/1000)
        interpolated_func = interp1d(original_time,all_speed, kind='linear')
        new_time = np.arange(0, int(original_time[-1]),1/down_to)
        self.RunSpeed = interpolated_func(new_time)

    # Read Temperature Last, as uncertain capture freq.
    def Temperature_Reader(self,capture_time,down_to =1):
        if 'FLIR' not in self.all_bio_rec:
            raise FileExistsError('Check whether have Body Temp Recording, or properly renamed')
        raw_temp_path = ot.Get_File_Name(ot.join(self.wp,'FLIR'),'.csv',include_sub=True)
        if len(raw_temp_path) != 1:
            raise LookupError('Number of Temperature File not right, check plz.')
        temperature_data = pd.read_csv(raw_temp_path[0], skiprows=3,header = None,encoding='latin-1')
        temperature_data = temperature_data.iloc[1:,1:-2].mean(1)

        temp_freq = len(temperature_data)/capture_time
        original_time = np.arange(len(temperature_data))/temp_freq
        interpolated_func = interp1d(original_time,temperature_data, kind='linear')
        new_time = np.arange(0, int(original_time[-1]),1/down_to)
        self.Body_Temperature = interpolated_func(new_time)

    def Do_Preprocess(self,down_freq = 1,time_from = 'EEG',eeg_method = 'plx'):

        self.All_Bio_Index = {}

        if 'EEG' in self.all_bio_rec:
            self.EEG_Reader(method=eeg_method)
            _ = self.EEG_Slide_Power(winsize = 60,step = 1/down_freq)
            if time_from == 'EEG':
                self.record_times = len(self.EEG_Signal)/self.EEG_fps
            self.All_Bio_Index['EEG'] = self.all_power

        if 'MouseOX' in self.all_bio_rec:
            self.MouseOX_Reader(down_to = down_freq)
            self.All_Bio_Index['Ox'] = self.OX_datas
            
        if 'Pad' in self.all_bio_rec:
            self.Pad_Reader(down_to = down_freq)
            self.All_Bio_Index['Pad_Movement'] = self.Pad_Movement
            if time_from == 'Pad':
                self.record_times = len(self.Pad_Movement)

        if 'Pupil' in self.all_bio_rec:
            self.Pupil_Reader(down_to = down_freq)
            self.All_Bio_Index['Pupil'] = self.Pupil

        if 'RunningSpeed' in self.all_bio_rec:
            self.Runspeed_Reader(down_to = down_freq)
            self.All_Bio_Index['RunSpeed'] = self.RunSpeed

        # At last, calculate temperature fps and intrapolate.
        if 'FLIR' in self.all_bio_rec:
            self.Temperature_Reader(capture_time = self.record_times,down_to = down_freq)
            self.All_Bio_Index['Bode_Temp'] = self.Body_Temperature

        ot.Save_Variable(self.save_path,'All_Bio_Paras_1Hz',self.All_Bio_Index)


#%% Test runs
if __name__ == '__main__':
    # wp = r'D:\#Shu_Data\TailPinch\ChatFlox_Cre\20240129_#243_chat-flox'
    # Okap = One_Key_Anes_Process(wp)
    # Okap.Do_Preprocess()
    # a = Okap.All_Bio_Index
    # all_folders = ot.Get_Subfolders(r'D:\#Shu_Data\TailPinch\ChatFlox_Cre','#')
    all_folders = ot.Get_Subfolders(r'D:\#Shu_Data\Data_SpontaneousWakeUp','#')
    # all_folders.pop(0) # as #243 done.
    # all_folders.pop(0) # as #244 done.
    # for i,c_path in enumerate(all_folders):
    c_path = all_folders[2]
    Okap = One_Key_Anes_Process(c_path)
    Okap.Do_Preprocess(time_from = 'Pad',eeg_method = 'smr')
    # a = ot.Load_Variable(ot.join(all_folders[-3],'_Processing'),'All_Bio_Paras_1Hz.pkl')
    # print(a['Ox'])
    a = Okap.All_Bio_Index['EEG']['All_EEG_Power']
    plt.plot(a)