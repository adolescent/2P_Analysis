# -*- coding: utf-8 -*-
"""
Created on Wed May 25 11:05:26 2022

@author: adolescent

Final version of caiman. This function solves the problem of files over 4GB.

"""
#%%
import os
from Decorators import Timer
import time
import Graph_Operation_Kit as gt
import OS_Tools_Kit as ot
import List_Operation_Kit as lt
import Caiman_API.Pack_Graphs as Graph_Packer
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
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import skimage.io
from caiman.source_extraction.cnmf.cnmf import load_CNMF

#%%

class One_Key_Caiman(object):
    
    name = r'Caiman Align and Cell Find.'
    # Boulder in UDLR sequense.
    def __init__(self,day_folder,run_lists,fps = 1.301,align_base = '1-003',boulder = (20,20,20,20),in_server = True,
                 max_shift = (75,75),align_batchsize = (100,100),align_overlap = (24,24),align_std = 3,
                 bk_comp_num = 2,rf = 20,k = 5,cnmf_overlap = 6,merge_thr = 0.85,snr = 2,rval_thr = 0.75,
                 min_cnn_thres = 0.90,cnn_lowest = 0.1,use_cuda = False,cut_size = 2000,n_process = 20,single_thread = True,decay = 1.4):
        
        self.day_folder = day_folder
        self.run_subfolders = lt.Run_Name_Producer_2P(run_lists)
        self.all_data_folders = lt.List_Annex([day_folder], self.run_subfolders)
        self.work_path = day_folder+r'\_CAIMAN'
        ot.mkdir(self.work_path)
        self.in_server = in_server
        #self.frame_lists = Graph_Packer(all_data_folders, self.work_path)
        self.fps = fps
        self.align_base = align_base
        self.boulder = boulder
        self.n_process = n_process
        self.single_thread = single_thread
        self.decay = decay
        # Check stack frame
        self.all_stack_names = ot.Get_File_Name(self.work_path)
        if self.all_stack_names == []:# if stack is unfinished
            print('Frame Stacks not Generated yet, stacking frames..')
            self.frame_lists,self.runname_dic = Graph_Packer.Graph_Packer_Cut(self.all_data_folders, self.work_path,cutsize = cut_size)
            self.all_stack_names = ot.Get_File_Name(self.work_path)
        else:
            print('Frame stacks already done.')
            self.frame_lists = Graph_Packer.Count_Frame_Num(self.all_data_folders)
            self.runname_dic = Graph_Packer.Get_Runname_Dic(run_lists, self.work_path)

        # Generate Parameter dictionary.
        opts_dict = {'fnames': self.all_stack_names,# Name list of all 
                     # dataset dependent parameters
                     'fr': self.fps,# Capture frequency
                     'decay_time': self.decay,# length of a typical transient in seconds
                     # motion correction parameters
                     'strides': align_batchsize,# start a new patch for pw-rigid motion correction every x pixels
                     'overlaps': align_overlap,# overlap between pathes (size of patch strides+overlaps)
                     'max_shifts': max_shift,# maximum allowed rigid shifts (in pixels)
                     'max_deviation_rigid': align_std, # maximum shifts deviation allowed for patch with respect to rigid shifts
                     'pw_rigid': True,# flag for performing non-rigid motion correction
                      # parameters for source extraction and deconvolution
                     'p': 1,# order of the autoregressive system
                     'nb': bk_comp_num,# number of global background components
                     'rf': rf,# half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
                     'K': k, # number of components per patch
                     'stride': cnmf_overlap,# amount of overlap between the patches in pixels
                     'method_init': 'greedy_roi',# initialization method (if analyzing dendritic data using 'sparse_nmf')
                     'rolling_sum': True,
                     'only_init': True,
                     'ssub': 1,# spatial subsampling during initialization
                     'tsub': 1,# temporal subsampling during intialization
                     'merge_thr': merge_thr, # merging threshold, max correlation allowed
                     'min_SNR': snr,# signal to noise ratio for accepting a component
                     'rval_thr':rval_thr,# space correlation threshold for accepting a component
                     'use_cnn': True,
                     'min_cnn_thr': min_cnn_thres,# threshold for CNN based classifier
                     'cnn_lowest': cnn_lowest,# neurons with cnn probability lower than this value are rejected
                     'use_cuda' : use_cuda # Set this to use cuda for motion correction.
                     }
        self.opts = params.CNMFParams(params_dict=opts_dict)
        
    def Motion_Corr_Single(self,c_runname,tamplate = None):
        
        used_filename = self.runname_dic[c_runname]
        if 'dview' in locals():
            cm.stop_server(dview=dview)
        #start a cluster for parallel processing (if a cluster already exists it will be closed and a new session will be opened)
        c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes= self.n_process, single_thread=self.single_thread)
        mc = MotionCorrect(used_filename, dview=dview , **self.opts.get_group('motion'))
        mc.motion_correct(template = tamplate,save_movie=True)
        border_to_0 = 0 if mc.border_nan == 'copy' else mc.border_to_0 
        # Save each memmap file.
        fname_each = cm.save_memmap_each(mc.mmap_file, base_name=c_runname.split('\\')[-1]+'_', order='C',border_to_0=border_to_0, dview=dview)
        #fname_new = cm.save_memmap(mc.mmap_file, base_name=c_runname.split('\\')[-1], order='C',border_to_0=border_to_0, dview=dview) # exclude borders
        fname_new = cm.save_memmap_join(fname_each)
        cm.stop_server(dview=dview)
        for i,c_mmap in enumerate(mc.mmap_file):
            os.remove(c_mmap)
        for i,c_cmmap in enumerate(fname_each):
            os.remove(c_cmmap)

        return fname_new
    
    def Motion_Corr_All(self):
        
        # get tamplate average first.
        self.all_avr_dic = {}
        tamplate_filename = self.Motion_Corr_Single(self.align_base)
        Yr, dims, T = cm.load_memmap(tamplate_filename)
        tamplate_series = np.reshape(Yr.T, [T] + list(dims), order='F') 
        tamplate = tamplate_series.mean(0)
        self.all_avr_dic[self.align_base] = tamplate
        clipped_tamplate = gt.Clip_And_Normalize(tamplate,clip_std = 5)
        gt.Show_Graph(clipped_tamplate, 'Align_Template', self.work_path)
        del Yr,dims,T,tamplate_series
        # Then align other runs.
        for i,c_run in enumerate(self.run_subfolders):
            if c_run != self.align_base:
                c_filename = self.Motion_Corr_Single(c_run,tamplate = tamplate)
                Yr, dims, T = cm.load_memmap(c_filename)
                c_series = np.reshape(Yr.T, [T] + list(dims), order='F') 
                c_avr = c_series.mean(0)
                self.all_avr_dic[c_run] = c_avr
                del Yr,dims,T,c_series
        # Then, we need to get global image file for cell find.
        all_mmap_name = ot.Get_File_Name(self.work_path,file_type = '.mmap')
        final_name = cm.save_memmap_join(all_mmap_name)
        for i,c_mmap in enumerate(all_mmap_name):
            os.remove(c_mmap)
        Yr, self.dims, T = cm.load_memmap(final_name)
        self.images = np.reshape(Yr.T, [T] + list(self.dims), order='F')
        # Plot global average. Save memory so we will not load images file here.
        ot.Save_Variable(self.work_path, 'Run_Averages', self.all_avr_dic)
        self.global_avr = np.zeros(shape = self.dims,dtype = 'f8')
        for i,c_run in enumerate(self.run_subfolders):
            c_frame_num = self.frame_lists[i]
            self.global_avr += self.all_avr_dic[c_run]*c_frame_num
        self.clipped_avr = gt.Clip_And_Normalize(self.global_avr,clip_std = 5)
        gt.Show_Graph(self.clipped_avr, 'Global_Average_cai', self.work_path)
        time.sleep(5)
        return True
        
    def Cell_Find(self,boulders):
        # Parrel process cost too much memory, here we use no par.
        # Try if we have images.
        print('Cell Finding...')
        try:
            self.images
        except AttributeError:
            c_mmap_name = ot.Get_File_Name(self.work_path,'.mmap')[0]
            print('Load mmap file from '+str(c_mmap_name))
            Yr, self.dims, T = cm.load_memmap(c_mmap_name)
            self.images = np.reshape(Yr.T, [T] + list(self.dims), order='F')
            del Yr,T
        # RUN CNMF ON PATCHES
        try:
            self.global_avr
        except AttributeError:
            print('No Global avr.')
            self.global_avr = self.images.mean(0)
            # self.global_avr = np.zeros(shape = self.dims,dtype = 'u1')
        self.cnm = cnmf.CNMF(20,params=self.opts)
        self.cnm = self.cnm.fit(self.images)
        # plot contours of found components
        self.cnm.estimates.plot_contours_nb(img=self.global_avr)
        # Refit to get real cell
        self.cnm2 = self.cnm.refit(self.images)
        self.cnm2.estimates.evaluate_components(self.images, self.cnm2.params)
        #self.cnm2.estimates.detrend_df_f(quantileMin=8, frames_window=250)
        self.cnm2.estimates.plot_contours_nb(img=self.global_avr, idx=self.cnm2.estimates.idx_components)
        
        # Wash cells, delete cell near boulder.
        self.real_cell_ids = []
        y_range = (boulders[0],self.dims[0]-boulders[1])
        x_range = (boulders[2],self.dims[1]-boulders[3])
        for i,c_comp in enumerate(self.cnm2.estimates.idx_components):
            c_loc = self.cnm2.estimates.coordinates[c_comp]['CoM']
            if (c_loc[1]>x_range[0] and c_loc[1]<x_range[1]) and (c_loc[0]>y_range[0] and c_loc[0]<y_range[1]):
                self.real_cell_ids.append(c_comp)
        self.cnm2.save(self.work_path+r'\analysis_results.hdf5')
        comp_id_dict = {}
        for i,cc in enumerate(self.real_cell_ids):
            comp_id_dict[i+1] = cc
        ot.Save_Variable(self.work_path, 'Component_ID_Lists', comp_id_dict)
    
        # Plot cell graph, both raw and labled.
        # Raw graph
        annotated_graph = np.zeros(shape = (512,512,3),dtype = 'f8')
        for i,c_id in enumerate(self.real_cell_ids):
            c_cell = np.reshape(self.cnm2.estimates.A[:,c_id].toarray(), (512,512), order='F')*100
            annotated_graph[:,:,0] += c_cell
            annotated_graph[:,:,1] += c_cell
            annotated_graph[:,:,2] += c_cell
        annotated_graph = gt.Clip_And_Normalize(annotated_graph,clip_std = 8,bit = 'u1')
        gt.Show_Graph(annotated_graph, 'Cell_Location', self.work_path)
        # Annotated graph
        font = ImageFont.truetype('arial.ttf',11)
        im = Image.fromarray(annotated_graph)
        for i,c_id in enumerate(self.real_cell_ids):
            y,x = self.cnm2.estimates.coordinates[c_id]['CoM']
            draw = ImageDraw.Draw(im)
            draw.text((x+5,y+5),str(i+1),(0,255,100),font = font,align = 'center')
        final_graph = np.array(im)
        gt.Show_Graph(final_graph, 'Numbers', self.work_path)
        
# =============================================================================
#     def Series_Generator_Manual(self):
#         self.cnm2.estimates.plot_contours_nb(img=self.global_avr, idx=self.real_cell_ids)
#         self.cell_series_dic = {}
#         for i,cc in tqdm(enumerate(self.real_cell_ids)):
#             self.cell_series_dic[i+1] = {}
#             # Annotate cell location in graph. Sequence X,Y.
#             self.cell_series_dic[i+1]['Cell_Loc'] = self.cnm2.estimates.coordinates[cc]['CoM']
#             c_mask = np.reshape(self.cnm2.estimates.A[:,cc].toarray(), self.dims, order='F')
#             self.cell_series_dic[i+1]['Cell_Mask'] = c_mask/c_mask.sum()
#             cc_series_all = (self.images*c_mask/c_mask.sum()).sum(axis = (1,2))
#             # cut series in different runs.
#             frame_counter = 0
#             for j,c_run in enumerate(self.run_subfolders):
#                 c_frame_num = self.frame_lists[j]
#                 self.cell_series_dic[i+1][c_run] = cc_series_all[frame_counter:frame_counter+c_frame_num]
#                 frame_counter+=c_frame_num
#         ot.Save_Variable(self.work_path, 'All_Series_Dic', self.cell_series_dic)
# =============================================================================
    def Series_Generator_Server(self):
        
        self.cnm2.estimates.plot_contours_nb(img=self.global_avr, idx=self.real_cell_ids)
        self.cell_series_dic = {}
        # get cell location mask.
        for i,cc in enumerate(self.real_cell_ids):
            self.cell_series_dic[i+1] = {}
            # Annotate cell location in graph. Sequence Y,X.
            self.cell_series_dic[i+1]['Cell_Loc'] = self.cnm2.estimates.coordinates[cc]['CoM']
            c_mask = np.reshape(self.cnm2.estimates.A[:,cc].toarray(), self.dims, order='F')
            self.cell_series_dic[i+1]['Cell_Mask'] = c_mask/c_mask.sum()
        # ot.Save_Variable(self.work_path,'Cell_Masks', self.cell_series_dic)
        # get cell response
        total_frame_num  = self.images.shape[0]
        cell_num = len(self.real_cell_ids)
        all_cell_data = np.zeros(shape = (cell_num,total_frame_num),dtype = 'f8')
        # A compromise between memory and speed.
        #for i,cc in tqdm(enumerate(self.real_cell_ids)):
        group_step = 50000
        group_num = np.ceil(total_frame_num/group_step).astype('int')
        #c_mask = np.reshape(self.cnm2.estimates.A[:,cc].toarray(), self.dims, order='F')
        for i in tqdm(range(group_num)):
            if i != group_num-1:# not the last group
                c_frame_group = np.array(self.images[i*group_step:(i+1)*group_step,:,:])
                for j,cc in tqdm(enumerate(self.real_cell_ids)):
                    c_mask = self.cell_series_dic[j+1]['Cell_Mask']
                    cc_resp = (c_frame_group*c_mask).sum(axis = (1,2))
                    all_cell_data[j,i*group_step:(i+1)*group_step] = cc_resp
                del c_frame_group
            else:# the last group
                c_frame_group = np.array(self.images[i*group_step:,:,:])
                for j,cc in tqdm(enumerate(self.real_cell_ids)):
                    c_mask = self.cell_series_dic[j+1]['Cell_Mask']
                    cc_resp = (c_frame_group*c_mask).sum(axis = (1,2))
                    all_cell_data[j,i*group_step:] = cc_resp 
                del c_frame_group
        # cut series in different runs.
        for i,cc in enumerate(self.real_cell_ids):
            frame_counter = 0
            cc_series_all = all_cell_data[i,:]
            for j,c_run in enumerate(self.run_subfolders):
                c_frame_num = self.frame_lists[j]
                self.cell_series_dic[i+1][c_run] = cc_series_all[frame_counter:frame_counter+c_frame_num]
                frame_counter+=c_frame_num
        ot.Save_Variable(self.work_path, 'All_Series_Dic', self.cell_series_dic)
        




    def Series_Generator_Low_Memory(self):
        
        self.cnm2.estimates.plot_contours_nb(img=self.global_avr, idx=self.real_cell_ids)
        self.cell_series_dic = {}
        # get cell location mask.
        for i,cc in enumerate(self.real_cell_ids):
            self.cell_series_dic[i+1] = {}
            # Annotate cell location in graph. Sequence X,Y.
            self.cell_series_dic[i+1]['Cell_Loc'] = self.cnm2.estimates.coordinates[cc]['CoM']
            c_mask = np.reshape(self.cnm2.estimates.A[:,cc].toarray(), self.dims, order='F')
            self.cell_series_dic[i+1]['Cell_Mask'] = c_mask/c_mask.sum()
        # ot.Save_Variable(self.work_path,'Cell_Masks', self.cell_series_dic)
        # get cell response
        total_frame_num  = self.images.shape[0]
        cell_num = len(self.real_cell_ids)
        all_cell_data = np.zeros(shape = (cell_num,total_frame_num),dtype = 'f8')
        # A compromise between memory and speed.
        #for i,cc in tqdm(enumerate(self.real_cell_ids)):
        group_step = 6000
        group_num = np.ceil(total_frame_num/group_step).astype('int')
        
        #c_mask = np.reshape(self.cnm2.estimates.A[:,cc].toarray(), self.dims, order='F')
        for i in tqdm(range(group_num)):
            if i != group_num-1:# not the last group
                c_frame_group = np.array(self.images[i*group_step:(i+1)*group_step,:,:])
                for j,cc in tqdm(enumerate(self.real_cell_ids)):
                    c_mask = self.cell_series_dic[j+1]['Cell_Mask']
                    cc_resp = (c_frame_group*c_mask).sum(axis = (1,2))
                    all_cell_data[j,i*group_step:(i+1)*group_step] = cc_resp
                del c_frame_group
            else:# the last group
                c_frame_group = np.array(self.images[i*group_step:,:,:])
                for j,cc in tqdm(enumerate(self.real_cell_ids)):
                    c_mask = self.cell_series_dic[j+1]['Cell_Mask']
                    cc_resp = (c_frame_group*c_mask).sum(axis = (1,2))
                    all_cell_data[j,i*group_step:] = cc_resp 
                del c_frame_group
                             
        # cut series in different runs.
        for i,cc in enumerate(self.real_cell_ids):
            frame_counter = 0
            cc_series_all = all_cell_data[i,:]
            for j,c_run in enumerate(self.run_subfolders):
                c_frame_num = self.frame_lists[j]
                self.cell_series_dic[i+1][c_run] = cc_series_all[frame_counter:frame_counter+c_frame_num]
                frame_counter+=c_frame_num
        ot.Save_Variable(self.work_path, 'All_Series_Dic', self.cell_series_dic)
        

    def Series_Generator_NG(self):# Use weighted sum to replace matric multiplication, hoping better performance.
        self.cnm2.estimates.plot_contours_nb(img=self.global_avr, idx=self.real_cell_ids)
        self.cell_series_dic = {}
        # get cell location mask.
        for i,cc in enumerate(self.real_cell_ids):
            self.cell_series_dic[i+1] = {}
            # Annotate cell location in graph. Sequence X,Y.
            self.cell_series_dic[i+1]['Cell_Loc'] = self.cnm2.estimates.coordinates[cc]['CoM']
            c_mask = np.reshape(self.cnm2.estimates.A[:,cc].toarray(), self.dims, order='F')
            self.cell_series_dic[i+1]['Cell_Mask'] = c_mask/c_mask.sum()
        # ot.Save_Variable(self.work_path,'Cell_Masks', self.cell_series_dic)
        # get cell response
        total_frame_num  = self.images.shape[0]
        cell_num = len(self.real_cell_ids)
        all_cell_data = np.zeros(shape = (cell_num,total_frame_num),dtype = 'f8')
        # cycle all cells to get series.
        print('Generating Cell F Trains...')
        all_frame = self.images.reshape(total_frame_num,-1)
        for i in tqdm(range(cell_num)):
            # get mask location and mask weight
            c_mask = self.cell_series_dic[i+1]['Cell_Mask'].flatten()
            c_mask_area = np.where(c_mask!=0)[0]
            c_weight = c_mask[c_mask_area] 
            # take out only mask weeight.
            c_train = np.dot(all_frame[:,c_mask_area],c_weight)
            all_cell_data[i,:] = c_train
        # cut series in different runs.
        for i,cc in enumerate(self.real_cell_ids):
            frame_counter = 0
            cc_series_all = all_cell_data[i,:]
            for j,c_run in enumerate(self.run_subfolders):
                c_frame_num = self.frame_lists[j]
                self.cell_series_dic[i+1][c_run] = cc_series_all[frame_counter:frame_counter+c_frame_num]
                frame_counter+=c_frame_num
        ot.Save_Variable(self.work_path, 'All_Series_Dic', self.cell_series_dic)            
            
                
        
        
    @Timer 
    def Do_Caiman(self):
        print('Motion correcting...')
        self.Motion_Corr_All()
        print('Cell_Finding')
        self.Cell_Find(boulders= self.boulder)
        print('Series_Generating...')
        if self.in_server:
            # self.Series_Generator_Low_Memory()
            self.Series_Generator_NG()
        else:
            print('Warning: memory might not enough.')
            self.Series_Generator_NG()
        
    
        
        
#%% Test run part.       
if __name__ == '__main__' :
    day_folder = r'D:\ZR\_Temp_Data\222222_L76_Test'
    run_lists = [4,5,6]
    plt.switch_backend('webAgg')
    Okc = One_Key_Caiman(day_folder, run_lists,align_base = '1-004',boulder = (20,20,20,20))
    #Okc.Do_Caiman()
    Okc.Motion_Corr_All()
    #Okc.global_avr = cv2.imread(r'G:\Test_Data\2P\220630_L76_2P\_CAIMAN\Summarize\Global_Average_cai.tif',-1)
    Okc.Cell_Find(boulders= Okc.boulder)
    # Okc.Series_Generator_Low_Memory()
    Okc.Series_Generator_NG()
