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

#%% F2, easy Z refilter

def Z_refilter(ac,start,end,runname = '1-001',HP=0.005,LP=False,clip_value=10,fps = 1.301):
    z_frame = np.zeros(shape = (len(ac),end-start),dtype='f8')
    for i,cc in tqdm(enumerate(ac.acn)):
        c_r = ac.all_cell_dic[cc]['1-001'][start:end]
        filted = Signal_Filter_v2(c_r,HP,LP,fps)
        dff_train = (filted-filted.mean())/filted.mean()
        z_train = dff_train/dff_train.std()
        z_frame[i,:] = z_train.clip(-clip_value,clip_value)
    return z_frame

def dff_refilter():
    pass