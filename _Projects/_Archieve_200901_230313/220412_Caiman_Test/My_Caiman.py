# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 15:42:46 2022

@author: ZR

Test Caiman on our data, test good threshold at the same time.
"""

import Graph_Operation_Kit as gt
import bokeh.plotting as bpl
from tqdm import tqdm
import cv2
import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params
from caiman.utils.utils import download_demo
from caiman.utils.visualization import plot_contours, nb_view_patches, nb_plot_contour



#%% Set log
logging.basicConfig(format="%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s] [%(process)d] %(message)s",
                    filename="/tmp/caiman.log",
                    level=logging.WARNING)
#%% Load in data.
import OS_Tools_Kit as ot
day_folder = r'G:\Test_Data\2P\222222_L76_Fake_Data_For_Caiman\1-005'
all_tif_name = ot.Get_File_Name(day_folder)
import tifffile as tif
image = np.zeros((len(all_tif_name),512,512), dtype='u2')
for i in range(len(all_tif_name)):
    c_graph = cv2.imread(all_tif_name[i],-1)
    image[i,:,:] = c_graph
    
tif.imwrite(r'G:\Test_Data\2P\222222_L76_Fake_Data_For_Caiman\Tiff_Stack.tif', image)

#%% Display movie before align.
repacked_filename = [r'G:\Test_Data\2P\222222_L76_Fake_Data_For_Caiman\Tiff_Stack.tif']
display_movie = True
if display_movie:
    m_orig = cm.load_movie_chain(repacked_filename,outtype=np.float32) # This have problem on u16 data..
    ds_ratio = 0.2
    m_orig.resize(1, 1, ds_ratio).play(q_max=99.5, fr=30, magnification=2)
#%% Setup some parameters.
# dataset dependent parameters
fr = 1.301                             # imaging rate in frames per second
decay_time = 2.5                    # length of a typical transient in seconds

# motion correction parameters
strides = (48, 48)          # start a new patch for pw-rigid motion correction every x pixels
overlaps = (24, 24)         # overlap between pathes (size of patch strides+overlaps)
max_shifts = (6,6)          # maximum allowed rigid shifts (in pixels)
max_deviation_rigid = 3     # maximum shifts deviation allowed for patch with respect to rigid shifts
pw_rigid = True             # flag for performing non-rigid motion correction

# parameters for source extraction and deconvolution
p = 1                       # order of the autoregressive system
gnb = 2                     # number of global background components
merge_thr = 0.85            # merging threshold, max correlation allowed
rf = 15                     # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
stride_cnmf = 6             # amount of overlap between the patches in pixels
K = 4                       # number of components per patch
gSig = [4, 4]               # expected half size of neurons in pixels
method_init = 'greedy_roi'  # initialization method (if analyzing dendritic data using 'sparse_nmf')
ssub = 1                    # spatial subsampling during initialization
tsub = 1                    # temporal subsampling during intialization

# parameters for component evaluation
min_SNR = 2.0               # signal to noise ratio for accepting a component
rval_thr = 0.85              # space correlation threshold for accepting a component
cnn_thr = 0.99              # threshold for CNN based classifier
cnn_lowest = 0.1 # neurons with cnn probability lower than this value are rejected
#%% Create a parameters object
opts_dict = {'fnames': repacked_filename,
            'fr': fr,
            'decay_time': decay_time,
            'strides': strides,
            'overlaps': overlaps,
            'max_shifts': max_shifts,
            'max_deviation_rigid': max_deviation_rigid,
            'pw_rigid': pw_rigid,
            'p': p,
            'nb': gnb,
            'rf': rf,
            'K': K, 
            'stride': stride_cnmf,
            'method_init': method_init,
            'rolling_sum': True,
            'only_init': True,
            'ssub': ssub,
            'tsub': tsub,
            'merge_thr': merge_thr, 
            'min_SNR': min_SNR,
            'rval_thr': rval_thr,
            'use_cnn': True,
            'min_cnn_thr': cnn_thr,
            'cnn_lowest': cnn_lowest}

opts = params.CNMFParams(params_dict=opts_dict)
#%% start a cluster for parallel processing (if a cluster already exists it will be closed and a new session will be opened)
if 'dview' in locals():
    cm.stop_server(dview=dview)
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)
#%% Motion correction
# first we create a motion correction object with the parameters specified
mc = MotionCorrect(repacked_filename, dview=dview, **opts.get_group('motion'),use_cuda=True)
# note that the file is not loaded in memory
#Now perform correction.

# Run piecewise-rigid motion correction using NoRMCorre
start_time = time.time()
mc.motion_correct(save_movie=True)
stop_time = time.time()
print('Motion Correction Cost:'+str(stop_time-start_time)+'s.')
m_els = cm.load(mc.fname_tot_els)
border_to_0 = 0 if mc.border_nan == 'copy' else mc.border_to_0 
    # maximum shift to be used for trimming against NaNs
#%% compare with original movie
display_movie = True
if display_movie:
    m_orig = cm.load_movie_chain(repacked_filename)
    ds_ratio = 0.2
    cm.concatenate([m_orig.resize(1, 1, ds_ratio) - mc.min_mov*mc.nonneg_movie,
                    m_els.resize(1, 1, ds_ratio)], 
                   axis=2).play(fr=60, gain=1, magnification=2, offset=0)  # press q to exit

#%%Memory mapping, save file in memory.
# memory map the file in order 'C'
fname_new = cm.save_memmap(mc.mmap_file, base_name='memmap_', order='C',
                           border_to_0=border_to_0, dview=dview) # exclude borders

# now load the file
Yr, dims, T = cm.load_memmap(fname_new)
images = np.reshape(Yr.T, [T] + list(dims), order='F') 
    #load frames in python format (T x X x Y)
#%% restart cluster to clean up memory
cm.stop_server(dview=dview)
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)
#%% RUN CNMF ON PATCHES
%%capture
# First extract spatial and temporal components on patches and combine them
# for this step deconvolution is turned off (p=0). If you want to have
# deconvolution within each patch change params.patch['p_patch'] to a
# nonzero value
cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
cnm = cnm.fit(images)
#%% plot contours of found components
Cn = cm.local_correlations(images.transpose(1,2,0))
Cn[np.isnan(Cn)] = 0
cnm.estimates.plot_contours_nb(img=Cn)
#%% RE-RUN seeded CNMF on accepted patches to refine and perform deconvolution 
cnm2 = cnm.refit(images, dview=dview)
#%% COMPONENT EVALUATION
# the components are evaluated in three ways:
#   a) the shape of each component must be correlated with the data
#   b) a minimum peak SNR is required over the length of a transient
#   c) each shape passes a CNN based classifier
cnm2.estimates.evaluate_components(images, cnm2.params, dview=dview)
#%% PLOT COMPONENTS
cnm2.estimates.plot_contours_nb(img=Cn, idx=cnm2.estimates.idx_components)
# accepted components
cnm2.estimates.nb_view_components(img=Cn, idx=cnm2.estimates.idx_components)
# rejected components
if len(cnm2.estimates.idx_components_bad) > 0:
    cnm2.estimates.nb_view_components(img=Cn, idx=cnm2.estimates.idx_components_bad)
else:
    print("No components were rejected.")
    
    #%% Extract DF/F values
cnm2.estimates.detrend_df_f(quantileMin=8, frames_window=250)
cnm2.estimates.nb_view_components(img=Cn, denoised_color='red')
print('you may need to change the data rate to generate this one: use jupyter notebook --NotebookApp.iopub_data_rate_limit=1.0e10 before opening jupyter notebook')
## Closing, saving, and creating denoised version
### You can save an hdf5 file with all the fields of the cnmf object
save_results = True
if save_results:
    cnm2.save('analysis_results.hdf5')
    
#%% STOP CLUSTER and clean up log files
cm.stop_server(dview=dview)
log_files = glob.glob('*_LOG_*')
for log_file in log_files:
    os.remove(log_file)
#%%
cnm2.estimates.play_movie(images, q_max=99.9, gain_res=2,
                                  save_movie = True
                                  magnification=2,
                                  bpx=border_to_0,
                                  include_bck=False)
#%% reconstruct denoised movie
denoised = cm.movie(cnm2.estimates.A.dot(cnm2.estimates.C) + \
                    cnm2.estimates.b.dot(cnm2.estimates.f)).reshape(dims + (-1,), order='F').transpose([2, 0, 1])
    
cnm2.estimates.play_movie(denoised)
