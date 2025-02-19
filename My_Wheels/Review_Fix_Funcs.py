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

def Z_refilter(ac,runname = '1-001',start=0,end=99999,HP=0.005,LP=False,clip_value=10,fps = 1.301):
    end = min(len(ac.all_cell_dic[1][runname]),end)
    z_frame = np.zeros(shape = (len(ac),end-start),dtype='f8')
    for i,cc in tqdm(enumerate(ac.acn)):
        c_r = ac.all_cell_dic[cc][runname][start:end]
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

def Rand_Series(event_num,series_length):

    total_zeros = series_length - event_num
    mandatory_zeros_between = event_num - 1
    extra_zeros_needed = total_zeros - mandatory_zeros_between
    def sample_stars_and_bars(stars, bins):
        if bins <= 1:
            return [stars]
        dividers = sorted(np.random.choice(stars + bins - 1, bins - 1, replace=False))
        gaps = []
        prev = -1
        for d in dividers:
            gaps.append(d - prev - 1)
            prev = d
        gaps.append(stars + bins - 1 - prev - 1)
        return gaps
    gaps = sample_stars_and_bars(extra_zeros_needed, event_num + 1)

    series = []
    # Add pre-gap zeros
    series.extend([0] * gaps[0])
    series.append(1)

    # Add internal gaps and ones
    for i in range(1, event_num):
        internal_zeros = gaps[i] + 1  # Mandatory 1 + extra
        series.extend([0] * internal_zeros)
        series.append(1)

    # Add post-gap zeros
    series.extend([0] * gaps[-1])
    series = np.array(series)
    return series


def Burstiness_Index_elife(series,N_shuffle=50):
    # still, series must be 1/0 series.
    event_num = int(series.sum())
    series_length = len(series)
    real_duration = np.diff(np.where(series==1)[0])
    cv_r = real_duration.std()/real_duration.mean()
    cv_s = np.zeros(N_shuffle)
    for i in range(N_shuffle):
        series_s = Rand_Series(event_num,series_length)
        shuffle_duration = np.diff(np.where(series_s==1)[0])
        cv_s[i] = (shuffle_duration.std())/shuffle_duration.mean()
    cv_ms = np.sum((cv_s-cv_s.mean())**2)/N_shuffle
    cv_s_mean = cv_s.mean()
    # here comes a bug, this function will lead to negative in sqrt = =
    bi = (cv_r-cv_s.mean())/np.sqrt(abs(cv_ms-cv_s_mean**2))
    return bi

def Burstiness_Index_JN(series,fraq = 0.15,winnum=300):
    event_num = int(series.sum())
    fraq_num = int(fraq*winnum)
    series_length = len(series)
    # real_duration = np.diff(np.where(series==1)[0])
    winlen = series_length//winnum
    win_response = np.reshape(series[:winnum*winlen],(winnum,winlen)).sum(1)
    best_resp = np.sort(win_response)[-fraq_num:]
    # get top percentage response's prop.
    f_best = best_resp.sum()/event_num
    bi = (f_best-fraq)/(1-fraq)
    return bi


#%% F4, Colorbar Generator
import matplotlib.colors as mcolors
import matplotlib as mpl
import seaborn as sns

def Cbar_Generate(vmin,vmax,center=0,cmap='bwr',figsize=(2,1),labelsize=8,aspect=10,shrink=1,dpi=600,orientation='horizontal'):
    data = [[vmin,vmax],[vmin,vmax]]
    # Create a heatmap
    fig, ax = plt.subplots(figsize = figsize,dpi = 600)
    g = sns.heatmap(data, cmap=cmap, center=center,ax = ax,vmax = vmax,vmin = vmin,cbar_kws={"aspect": aspect,"shrink": shrink,"orientation": orientation})
    ax.set_visible(False)
    g.collections[0].colorbar.set_ticks([vmin,center,vmax])
    g.collections[0].colorbar.set_ticklabels([vmin,center,vmax])
    g.collections[0].colorbar.ax.tick_params(labelsize=8)
    # g.collections[0].colorbar.aspect(50)
    # Create colorbar
    # fig.colorbar(ax2.collections[0], ax=ax, orientation='vertical')
    fig.tight_layout()
    return fig,ax 