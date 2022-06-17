# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 16:50:07 2022

@author: ZR
"""

import os
from Decorators import Timer
import time
import Graph_Operation_Kit as gt
import OS_Tools_Kit as ot
import List_Operation_Kit as lt
from Caiman_API.Pack_Graphs import Graph_Packer
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



#%%
class One_Key_Caiman(object):
    
    name = r'Caiman operation '
    # Boulder in range Up,Down,Left,Right.
    def __init__(self,day_folder,run_lists,fps = 1.301,align_base = '1-003',boulder = (20,20,20,20)):
        self.day_folder = day_folder
        self.run_subfolders = lt.Run_Name_Producer_2P(run_lists)
        self.all_data_folders = lt.List_Annex([day_folder], self.run_subfolders)
        self.work_path = day_folder+r'\_CAIMAN'
        ot.mkdir(self.work_path)
        #self.frame_lists = Graph_Packer(all_data_folders, self.work_path)
        self.fps = fps
        self.align_base = align_base
        self.boulder = boulder
        
    def Pack_Graphs(self):
        self.frame_lists = Graph_Packer(self.all_data_folders, self.work_path)
        
    def Parameter_Initial(self):
        self.all_stack_names = ot.Get_File_Name(self.work_path)
        opts_dict = {'fnames': self.all_stack_names,# Name list of all 
                     # dataset dependent parameters
                     'fr': self.fps,# Capture frequency
                     'decay_time': 2,# length of a typical transient in seconds
                     # motion correction parameters
                     'strides': (100,100),# start a new patch for pw-rigid motion correction every x pixels
                     'overlaps': (24, 24),# overlap between pathes (size of patch strides+overlaps)
                     'max_shifts': (50,50),# maximum allowed rigid shifts (in pixels)
                     'max_deviation_rigid': 3, # maximum shifts deviation allowed for patch with respect to rigid shifts
                     'pw_rigid': True,# flag for performing non-rigid motion correction
                      # parameters for source extraction and deconvolution
                     'p': 1,# order of the autoregressive system
                     'nb': 3,# number of global background components
                     'rf': 25,# half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
                     'K': 6, # number of components per patch
                     'stride': 6,# amount of overlap between the patches in pixels
                     'method_init': 'greedy_roi',# initialization method (if analyzing dendritic data using 'sparse_nmf')
                     'rolling_sum': True,
                     'only_init': True,
                     'ssub': 1,# spatial subsampling during initialization
                     'tsub': 1,# temporal subsampling during intialization
                     'merge_thr': 0.85, # merging threshold, max correlation allowed
                     'min_SNR': 2,# signal to noise ratio for accepting a component
                     'rval_thr':0.85,# space correlation threshold for accepting a component
                     'use_cnn': True,
                     'min_cnn_thr': 0.99,# threshold for CNN based classifier
                     'cnn_lowest': 0.1,# neurons with cnn probability lower than this value are rejected
                     'use_cuda' : True # Set this to use cuda for motion correction.
                     }
        self.opts = params.CNMFParams(params_dict=opts_dict)
        
    def Motion_Correct(self):
        if 'dview' in locals():
            cm.stop_server(dview=dview)
        #start a cluster for parallel processing (if a cluster already exists it will be closed and a new session will be opened)
        c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)
        mc = MotionCorrect(self.all_stack_names, dview=dview , **self.opts.get_group('motion'))
        start_time = time.time()
        mc.motion_correct(save_movie=True)
        stop_time = time.time()
        print('Motion Correction Cost:'+str(stop_time-start_time)+'s.')
        border_to_0 = 0 if mc.border_nan == 'copy' else mc.border_to_0 
# =============================================================================
#         m_els = cm.load(mc.fname_tot_els)
#         display_movie = True
#         if display_movie:
#             m_orig = cm.load_movie_chain(self.all_stack_names)
#             ds_ratio = 0.1
#             cm.concatenate([m_orig.resize(1, 1, ds_ratio) - mc.min_mov*mc.nonneg_movie,
#                             m_els.resize(1, 1, ds_ratio)], 
#                            axis=2).play(fr=60, gain=1, magnification=2, offset=0,
#                                         save_movie = True,opencv_codec = 'MPGE',movie_name = self.work_path+r'\Align_Compare.mp4')  # press q to exit
# =============================================================================
        # Save corrected graph in mmap files.
        fname_new = cm.save_memmap(mc.mmap_file, base_name='memmap_', order='C',border_to_0=border_to_0, dview=dview) # exclude borders
        # Clost the cluster to release memory.
        cm.stop_server(dview=dview)
        # now load the file
        Yr, self.dims, T = cm.load_memmap(fname_new)
        self.images = np.reshape(Yr.T, [T] + list(self.dims), order='F') 
        # Plot global average
        global_avr = self.images.mean(0)
        self.clipped_avr = gt.Clip_And_Normalize(global_avr,clip_std = 5)
        gt.Show_Graph(self.clipped_avr, 'Global_Average_cai', self.work_path)
        
    def Motion_Correct_Single_File(self,c_runname,tamplate = None):
        # this is the core file of motion correction.
        used_filename = c_runname.split('\\')[-1][:-4]
        if 'dview' in locals():
            cm.stop_server(dview=dview)
        #start a cluster for parallel processing (if a cluster already exists it will be closed and a new session will be opened)
        c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)
        mc = MotionCorrect(c_runname, dview=dview , **self.opts.get_group('motion'))
        mc.motion_correct(template = tamplate,save_movie=True)
        border_to_0 = 0 if mc.border_nan == 'copy' else mc.border_to_0 
        fname_new = cm.save_memmap(mc.mmap_file, base_name=used_filename, order='C',border_to_0=border_to_0, dview=dview) # exclude borders
        cm.stop_server(dview=dview)
        os.remove(mc.mmap_file[0])
        # now load the file
# =============================================================================
#         Yr, dims, T = cm.load_memmap(fname_new)
#         crun_series = np.reshape(Yr.T, [T] + list(dims), order='F') 
#         run_averaged_graph = crun_series.mean(0)
# =============================================================================
        return fname_new
        
        
    def Motion_Correction_Low_Memory(self):
        
        all_stack_names = ot.Get_File_Name(self.work_path,'.tif')
        tamplate_runname = self.work_path+'\\'+self.align_base+'.tif'
        # get tamplate average first.
        tamplate_filename = self.Motion_Correct_Single_File(tamplate_runname)
        Yr, dims, T = cm.load_memmap(tamplate_filename)
        tamplate_series = np.reshape(Yr.T, [T] + list(dims), order='F') 
        tamplate = tamplate_series.mean(0)
        clipped_tamplate = gt.Clip_And_Normalize(tamplate,clip_std = 5)
        gt.Show_Graph(clipped_tamplate, 'Align_Template', self.work_path)
        del Yr,dims,T,tamplate_series
        # Then align other runs.
        for i,c_run in enumerate(all_stack_names):
            if c_run.split('\\')[-1][:-4] != self.align_base:
                self.Motion_Correct_Single_File(c_run,tamplate = tamplate)
        # Then, we need to get global image file for cell find.
        all_mmap_name = ot.Get_File_Name(self.work_path,file_type = '.mmap')
        final_name = cm.save_memmap_join(all_mmap_name)
        Yr, self.dims, T = cm.load_memmap(final_name)
        self.images = np.reshape(Yr.T, [T] + list(self.dims), order='F')
        # Plot global average
        global_avr = self.images.mean(0)
        self.clipped_avr = gt.Clip_And_Normalize(global_avr,clip_std = 5)
        gt.Show_Graph(self.clipped_avr, 'Global_Average_cai', self.work_path)
        time.sleep(300)
# =============================================================================
#         if len(all_mmap_name) == 1:
#             Yr, self.dims, T = cm.load_memmap(all_mmap_name[0])
#             self.images = np.reshape(Yr.T, [T] + list(self.dims), order='F') 
#         else:
#             Yr, self.dims, T = cm.load_memmap(all_mmap_name[0])
#             for i in range(1,len(all_mmap_name)):
#                 c_append_Yr, _, c_append_T = cm.load_memmap(all_mmap_name[i])
#                 Yr = np.concatenate((Yr,c_append_Yr),axis = 1)
#                 T += c_append_T
#         self.images = np.reshape(Yr.T, [T] + list(self.dims), order='F')
#         
# =============================================================================
        return True
        
        
        
    @Timer
    def Cell_Find(self):
        # restart cluster to clean up memory
        print('Start Cell Finding...')
        c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)
        # RUN CNMF ON PATCHES
        cnm = cnmf.CNMF(n_processes, params=self.opts, dview=dview)
        cnm = cnm.fit(self.images)
        # plot contours of found components
        self.Cn = cm.local_correlations(self.images.transpose(1,2,0))
        self.Cn[np.isnan(self.Cn)] = 0
        cnm.estimates.plot_contours_nb(img=self.Cn)
        self.cnm2 = cnm.refit(self.images, dview=dview)
        self.cnm2.estimates.evaluate_components(self.images, self.cnm2.params, dview=dview)
        self.cnm2.estimates.detrend_df_f(quantileMin=8, frames_window=250)
        self.cnm2.estimates.plot_contours_nb(img=self.Cn, idx=self.cnm2.estimates.idx_components)
        
        # Wash cells, delete cell near boulder.
        self.real_cell_ids = []
        y_range = (self.boulder[0],self.dims[0]-self.boulder[1])
        x_range = (self.boulder[2],self.dims[1]-self.boulder[3])
        for i,c_comp in enumerate(self.cnm2.estimates.idx_components):
            c_loc = self.cnm2.estimates.coordinates[c_comp]['CoM']
            if (c_loc[0]>x_range[0] and c_loc[0]<x_range[1]) and (c_loc[1]>y_range[0] and c_loc[1]<y_range[1]):
                self.real_cell_ids.append(c_comp)
            
        self.cnm2.save(self.work_path+r'\analysis_results.hdf5')
        
    def Series_Generator(self):
        # This part is used to generate able cell parts.
        self.cnm2.estimates.plot_contours_nb(img=self.Cn, idx=self.real_cell_ids)
        self.cell_series_dic = {}
        for i,cc in tqdm(enumerate(self.real_cell_ids)):
            self.cell_series_dic[i+1] = {}
            # Annotate cell location in graph. Sequence X,Y.
            self.cell_series_dic[i+1]['Cell_Loc'] = self.cnm2.estimates.coordinates[cc]['CoM']
            self.cell_series_dic[i+1]['Cell_Mask'] = np.reshape(self.cnm2.estimates.A[:,cc].toarray(), self.dims, order='F')>0
            cc_series_all = self.cnm2.estimates.F_dff[cc,:]
            # cut series in different runs.
            frame_counter = 0
            for j,c_run in enumerate(self.run_subfolders):
                c_frame_num = self.frame_lists[j]
                self.cell_series_dic[i+1][c_run] = cc_series_all[frame_counter:frame_counter+c_frame_num]
                frame_counter+=c_frame_num
        ot.Save_Variable(self.work_path, 'All_Series_Dic', self.cell_series_dic)
    
    def Plot_Necessary_Graphs(self):
        # This part is used to generate personal used parts of data.
        
        # Then plot counter graph.
        self.cnm2.estimates.plot_contours(img=self.clipped_avr, idx=self.real_cell_ids)
        graph_base = cv2.cvtColor(self.clipped_avr, cv2.COLOR_GRAY2BGR) 
        for i in range(len(self.cell_series_dic)):
            c_y,c_x = self.cell_series_dic[i+1]['Cell_Loc']
            cv2.circle(graph_base,(int(c_x),int(c_y)),radius = 5,color = (0,0,65535),thickness =1)
        gt.Show_Graph(graph_base, 'Annotated_Graph', self.work_path)
        
    @Timer
    def Do_Caiman_Calculation_Huge_Memory(self):
        # One key from align to cell find.
        # This only work is memory is large enough. In most time we have to use following ones.
        self.Pack_Graphs()
        self.Parameter_Initial()
        self.Motion_Correct()
        self.Cell_Find()
        self.Series_Generator()
        self.Plot_Necessary_Graphs()
        
    @Timer    
    def Do_Caiman_Calculation(self):
        self.Pack_Graphs()
        self.Parameter_Initial()
        self.Motion_Correction_Low_Memory()
        self.Cell_Find()
        self.Series_Generator()
        self.Plot_Necessary_Graphs()
        
 #%% Test run part.       
if __name__ == '__main__' :
    day_folder = r'D:\ZR\_Temp_Data\210721_L76_2P'
    run_lists = [3]
    Okc = One_Key_Caiman(day_folder, run_lists,align_base = '1-003',boulder = (25,25,25,25))
    Okc.Do_Caiman_Calculation()
# =============================================================================
#   Use this for debug.
#     Okc.Pack_Graphs()
#     Okc.Parameter_Initial()
#     Okc.Motion_Correction_Low_Memory()
#     Okc.Cell_Find()
#     Okc.Series_Generator()
#     Okc.Plot_Necessary_Graphs()
# =============================================================================
    #
    # Check memory using problems.
    