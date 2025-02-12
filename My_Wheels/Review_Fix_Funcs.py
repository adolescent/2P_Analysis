'''
Several core functions used for review fix.
Almost the same function as that in class, but more easy to use, and we have some api change.

'''
import numpy as np
import matplotlib.pyplot as plt
from Filters import Signal_Filter_v2
from tqdm import tqdm

#%%% F1, FFT for series

def FFT_Spectrum(series, fps,ticks = 0.01,plot = False):
    """
    Compute Power spectrum of given series, note that it will return power density and raw power at the same time.

    Parameters:
        series (array-like): Input time series data.
        fps (float): Sampling frequency (frames per second).
        ticks (float): Ticks of power spectrum.
        plot (Bool): Whether Plot power spectrum result.

    Returns:
        freq_ticks (ndarray): Array of frequencies.
        binned_power (ndarray): Power at each frequency band.
        freqs_raw (ndarray): Raw freq tick, this will be affected by series length.
        power_raw (ndarray): Raw power of FFT results, not normalized.
        total_power (float): Full power of current 
    """
    n = len(series)  # Number of data points

    # Compute the FFT
    fft_result = np.fft.fft(series)
    freqs_raw = np.fft.fftfreq(n, d=1/fps)  # Frequency bins

    # Compute the power spectral density (PSD)
    power_raw = np.abs(fft_result) ** 2 / (fps * n)  # Normalized PSD
    power_raw = power_raw[:n // 2]  # Keep only positive frequencies
    freqs_raw = freqs_raw[:n // 2]  # Keep only positive frequencies

    # Normalize the PSD so that the total power sums to 1
    total_power = np.sum(power_raw)
    power_density = power_raw / total_power


    # Bin the power density
    bin_edges = np.arange(0, freqs_raw[-1] + ticks, ticks)
    freq_ticks = (bin_edges[:-1] + bin_edges[1:]) / 2  # Center of each bin
    binned_power = np.zeros_like(freq_ticks)
    # sum power inside given bin band.
    for i in range(len(bin_edges) - 1):
        # Find frequencies within the current bin
        mask = (freqs_raw >= bin_edges[i]) & (freqs_raw < bin_edges[i + 1])
        binned_power[i] = np.sum(power_density[mask])

    # plot if required.
    if plot == True:
        plt.figure(figsize=(6,6))
        # plt.plot(freqs, power_density, label="Power Spectrum Raw")
        plt.bar(freq_ticks, binned_power, width=ticks, align='center', alpha=0.7, label="Binned PSD")
        # plt.plot(freq_ticks, binned_power, alpha=0.7, label="Binned PSD")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power Density")
        plt.title(f"Binned Power Spectral Density (Bin Width = {ticks} Hz)")
        # plt.grid(True)
        plt.legend()
        plt.show()

    return freq_ticks,binned_power,freqs_raw,power_raw,total_power

#%% F2, easy Z refilter & easy dff refilter

def Z_refilter(ac,start,end,runname = '1-001',HP=0.005,LP=False,clip_value=10,fps = 1.301):
    z_frame = np.zeros(shape = (len(ac),end-start),dtype='f8')
    for i,cc in tqdm(enumerate(ac.acn)):
        c_r = ac.all_cell_dic[cc]['1-001'][start:end]
        filted = Signal_Filter_v2(c_r,HP,LP,fps)
        dff_train = (filted-filted.mean())/filted.mean()
        z_train = dff_train/dff_train.std()
        z_frame[i,:] = z_train.clip(-clip_value,clip_value)
    return z_frame

def dff_refilter(ac,runname = '1-001',start=0,end=99999,HP=0.005,LP=False,clip_value=10,fps = 1.301,prop=0.1):

    # get f matrix first.
    end = min(len(ac.all_cell_dic[1][runname]),end)
    F_frames_all = np.zeros(shape = (end-start,len(ac)),dtype='f8')
    for i,cc in enumerate(ac.acn):
        c_series_raw = ac.all_cell_dic[cc][runname][start:end]
        c_series_filted = Signal_Filter_v2(c_series_raw,order=5,HP_freq=HP,LP_freq=LP,fps = fps)
        F_frames_all[:,i] = c_series_filted
    # then calculate dF/F matrix.
    dFF_matrix = np.zeros(shape = F_frames_all.shape,dtype='f8')
    for i in range(F_frames_all.shape[1]):
        c_F_series = F_frames_all[:,i]
        base_num = int(len(c_F_series)*prop)
        base_id = np.argpartition(c_F_series, base_num)[:base_num]
        base = c_F_series[base_id].mean()
        c_dff_series = (c_F_series-base)/base
        c_dff_series = np.clip(c_dff_series,-clip_value,clip_value)
        dFF_matrix[:,i] = c_dff_series

    return dFF_matrix

#%% F3, burstiness parameter.
def Burstiness_Index(series):
    # series MUST be an 1/0 series, 1 as event on and 0 as off.
    event_indices = np.where(np.array(series) == 1)[0]
    
    # Calculate the inter-event times (differences between consecutive 1s)
    inter_event_times = np.diff(event_indices)
    
    # If there are no 1s or only one 1, burstiness is undefined
    if len(inter_event_times) == 0:
        return 0  # or np.nan, depending on how you want to handle this case
    
    # Calculate mean and variance of inter-event times
    mean_inter_event = np.mean(inter_event_times)
    std_inter_event = np.std(inter_event_times)
    
    # Calculate Burstiness Index
    if mean_inter_event == 0 and std_inter_event == 0:
        return 0  # or np.nan, if all events are consecutive
    burstiness_index = (std_inter_event - mean_inter_event) / (std_inter_event + mean_inter_event)
    
    return burstiness_index

#%% F4, Colorbar Generator
import matplotlib.colors as mcolors
import matplotlib as mpl

def Cbar_Generate(vmax,vmin,cmap='bwr',figsize=(2,1),labelsize=8,aspect=10,shrink=1,dpi=600,orientation='horizontal'):
    data = [[vmin,vmax],[vmin,vmax]]
    # Create a heatmap
    fig, ax = plt.subplots(figsize = figsize,dpi = 600)

    return fig,ax 