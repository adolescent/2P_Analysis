# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 13:24:15 2021

@author: ZR
"""

import  Stim_Dic_Tools as SDT
import numpy as np
import OS_Tools_Kit as ot
import matplotlib.pyplot as plt
import cv2
import Graph_Operation_Kit as gt
import Statistic_Tools as st
import random
import seaborn as sns

class Cell_Processor(object):
    
    name = 'Cell Data Processor'
    
    def __init__(self,day_folder,average_graph = 'Default'):
        '''
        Cell Processor with given inputs.

        Parameters
        ----------
        day_folder : (str)
            Save folder of day data. All_Cell,All_Stim_Frame Align need to be in this file.
        average_graph : (2D Array,dtype = u16), optional
            Global average graph. Need to be 16bit gray graph. The default is 'Default',meaning this graph is read from root path.

        '''
        
        print('Cell data processor, makesure you have cell data.')
        cell_data_path = ot.Get_File_Name(day_folder,'.ac')[0]
        all_stim_dic_path = ot.Get_File_Name(day_folder,'.sfa')[0]
        self.all_cell_dic = ot.Load_Variable(cell_data_path)
        if average_graph == 'Default':
            self.average_graph = cv2.imread(day_folder+r'\Global_Average.tif',-1)
        else:
            self.average_graph = average_graph
        self.all_stim_dic = ot.Load_Variable(all_stim_dic_path)
        self.all_cell_names = list(self.all_cell_dic.keys())
        self.cell_num = len(self.all_cell_dic)
        self.save_folder = day_folder+r'\_All_Results'
        ot.mkdir(self.save_folder)
        
        
    def Cell_Response_Maps(self,runname,
                           Condition_dics,
                           mode = 'processed',
                           stim_on = (3,6),
                           error_bar = True,
                           figsize = 'Default',
                           subshape = 'Default'
                           ):
        '''
        Cell Response map generator

        Parameters
        ----------
        runname : (int)
            Run for plot. In format 'Run001'
        Condition_dics : (Dic)
            Condition-ID combiner. This can be generated from Stim_ID_Combiner.
        mode : 'processed' or 'raw', optional
            Determine wheter we use CR or Raw_CR train. The default is 'processed'.
        stim_on : (turple), optional
            Range of stim on. The default is (3,6).
        error_bar : bool, optional
            Whether we plot error bar on graph. The default is True.
        figsize : (turple), optional
            Size of figure. Only need for too many condition. The default is 'Default'.
        subshape : (turple), optional
            Shape of subgraph layout.Row*Colume. The default is 'Default'.


        '''
        graph_folder = self.save_folder+r'\\'+runname
        ot.mkdir(graph_folder)
        
        for i in range(self.cell_num):# all cells
            c_cellname = self.all_cell_names[i]
            tc = self.all_cell_dic[c_cellname]
            #Is this cell in run?
            if runname not in tc:
                print('Cell '+ c_cellname + ' Not in '+ runname)
                continue
            # Do we have CR train in this cell?
            if 'CR_Train' not in tc[runname]:
                print('Cell '+ c_cellname + ' have no respose data.')
                continue
            # get cr trains & plotable data. 
            if mode == 'processed':
                cr_train = tc[runname]['CR_Train']
            elif mode == 'raw':
                cr_train = tc[runname]['Raw_CR_Train']
            else:
                raise IOError('Wrong CR Mode.')
            plotable_data = SDT.CR_Train_Combiner(cr_train, Condition_dics)
            # Plot graphs.
            response_plot_dic = {}
            subgraph_num = len(plotable_data)
            all_subgraph_name = list(plotable_data.keys())
            y_max = 0# get y sticks
            y_min = 65535
            for j in range(subgraph_num):
                current_graph_response = plotable_data[all_subgraph_name[j]]
                average_plot = current_graph_response.mean(0)
                se_2 = current_graph_response.std(0)/np.sqrt(current_graph_response.shape[0])*2
                response_plot_dic[all_subgraph_name[j]] = (average_plot,se_2)
                # renew y min and y max.
                if average_plot.min() < y_min:
                    y_min = average_plot.min()
                if average_plot.max() > y_max:
                    y_max = average_plot.max()
            y_range = [y_min-0.3,y_max+0.3]
            # Graph Plotting
            if subshape == 'Default':
                col_num = int(np.ceil(np.sqrt(subgraph_num)))
                row_num = int(np.ceil(subgraph_num/col_num))
            else:
                col_num = subshape[1]
                row_num = subshape[0]
            if figsize == 'Default':
                fig,ax = plt.subplots(row_num,col_num,figsize = (15,15))# Initialize graphs:
            else:
                fig,ax = plt.subplots(row_num,col_num,figsize = figsize)
            fig.suptitle(c_cellname+'_Response Maps', fontsize=30)
            for j in range(subgraph_num):
                current_col = j%col_num
                current_row = j//col_num
                current_graph_name = all_subgraph_name[j]
                current_data = response_plot_dic[current_graph_name]
                frame_num = len(current_data[0])
                # Start plot
                ax[current_row,current_col].hlines(y_range[0]+0.05, stim_on[0],stim_on[1],color="r")
                ax[current_row,current_col].set_ylim(y_range)
                ax[current_row,current_col].set_xticks(range(frame_num))
                ax[current_row,current_col].set_title(current_graph_name)
                # Whether we plot error bar on graph.
                if error_bar == True:
                    ax[current_row,current_col].errorbar(range(frame_num),current_data[0],current_data[1],fmt = 'bo-',ecolor='g')
                else:
                    ax[current_row,current_col].errorbar(range(frame_num),current_data[0],fmt = 'bo-')
            # Save ploted graph.
            ot.Save_Variable(graph_folder, c_cellname+'_Response_Data', plotable_data)
            fig.savefig(graph_folder+r'\\'+c_cellname+'_Response.png',dpi = 180)
            plt.clf()
            plt.close()
        return True
            
    def Radar_Maps(self,runname,
                   Radar_Cond,
                   on_frames = [3,4,5,6],
                   bais_angle = 0,
                   mode = 'processed',
                   error_bar = True):
        '''
        Generate radar map of given tuning properties. Not all stim can draw this.

        Parameters
        ----------
        runname : (str)
            Run we use. In format 'Run001'
        Radar_Cond : (Dic)
            Dictionary of condition for radar plot. This is generated by 'Stim_ID_Combiner'.
        on_frames : (list), optional
            Stim On frames. Use max response of this as reaction. ROI can be different.The default is [3,4,5,6].
        bais_angle : (float), optional
            Turning angle of axis, anti-clock wise. The default is 0.
        mode : ('processed' or 'raw'), optional
            CR or Raw_CR. The default is 'processed'.
        error_bar : bool, optional
            Whether we plot error bar on graph. The default is True.

        Raises
        ------
        IOError
            DESCRIPTION.

        Returns
        -------
        bool
            DESCRIPTION.

        '''
        radar_folder = self.save_folder+r'\\'+runname+'_Radar_Maps'
        ot.mkdir(radar_folder)
        for i in range(self.cell_num):# all cells
            c_cellname = self.all_cell_names[i]
            tc = self.all_cell_dic[c_cellname]
            #Is this cell in run?
            if runname not in tc:
                print('Cell '+ c_cellname + ' Not in '+ runname)
                continue
            # Do we have CR train in this cell?
            if 'CR_Train' not in tc[runname]:
                print('Cell '+ c_cellname + ' have no respose data.')
                continue
            # get cr trains & plotable data. 
            if mode == 'processed':
                cr_train = tc[runname]['CR_Train']
            elif mode == 'raw':
                cr_train = tc[runname]['Raw_CR_Train']
            else:
                raise IOError('Wrong CR Mode.')
            radar_data = SDT.CR_Train_Combiner(cr_train,Radar_Cond)
            all_radar_names = list(radar_data.keys())
            plotable_data = {}
            plotable_data['Names'] = []
            plotable_data['Values'] = np.zeros(len(all_radar_names),dtype = 'f8')
            plotable_data['Stds'] = np.zeros(len(all_radar_names),dtype = 'f8')
            for j in range(len(all_radar_names)):
                c_name = all_radar_names[j]
                plotable_data['Names'].append(c_name)
                c_conds,c_ses = radar_data[c_name].mean(0),radar_data[c_name].std(0)*2/np.sqrt(radar_data[c_name].shape[0])
                cutted_conds,cutted_std = c_conds[on_frames],c_ses[on_frames]
                max_ps = np.where(cutted_conds == cutted_conds.max())[0][0]
                plotable_data['Values'][j] = cutted_conds[max_ps]
                plotable_data['Stds'][j] = cutted_std[max_ps]
            # plot radar maps.
            fig = plt.figure(figsize = (8,8))
            fig.suptitle(c_cellname+'_Radar Maps',fontsize=22)
            ax = plt.axes(polar=True)
            ax.set_theta_zero_location("N")
            ax_num = len(all_radar_names)
            angle_series =2*np.pi/360 * (np.linspace(0, 360, ax_num+1,dtype = 'f8')+bais_angle)
            ax.set_xticks(angle_series[:-1])
            ax.set_xticklabels(plotable_data['Names'])
            if error_bar == True:
                ax.errorbar(angle_series, 
                            np.append(plotable_data['Values'],plotable_data['Values'][0]),
                            np.append(plotable_data['Stds'],plotable_data['Stds'][0]),
                            fmt = 'bo-',ecolor='r')
            else:
                ax.plot(angle_series, np.append(plotable_data['Values'],plotable_data['Values'][0]),'bo-')# Add one to close plots.
            # at last, save graphs.
            ot.Save_Variable(radar_folder, c_cellname+'_Radar_Data', plotable_data)
            fig.savefig(radar_folder+r'\\'+c_cellname+'_Radar.png',dpi = 180)
            plt.clf()
            plt.close()
        return True
            
    def Single_Cell_Plotter(self,cell_name,mode='circle',show_time = 5000):
        '''
        Generate Single Cell Location map.

        Parameters
        ----------
        cell_name : (str)
            Name of cell you want to plot,'self.all_cell_names' can help.
        mode : ('circle' or 'fill'), optional
            Method of plot. The default is 'circle'.


        '''
        cell_save_path = self.save_folder+r'\Single_Cells'
        ot.mkdir(cell_save_path)
        if type(self.average_graph) == type(None):
            raise IOError('We need Average graph to generate cell loc.')
        c_cell_dic = self.all_cell_dic[cell_name]
        c_cell_info = c_cell_dic['Cell_Info']
        # Then plot cells, in mode we want.
        if mode == 'fill':# meaning we will paint graph into green.
            base_graph = cv2.cvtColor(self.average_graph,cv2.COLOR_GRAY2RGB)*0.7
            cell_y,cell_x = c_cell_info.coords[:,0],c_cell_info.coords[:,1]
            base_graph[cell_y,cell_x,1] +=32768
            base_graph = np.clip(base_graph,0,65535).astype('u2')
        elif mode == 'circle':
            base_graph = cv2.cvtColor(self.average_graph,cv2.COLOR_GRAY2RGB)
            loc_y,loc_x = c_cell_info.centroid
            base_graph = cv2.circle(base_graph,(int(loc_x),int(loc_y)),radius = 10,color = (0,0,65535),thickness=2)
        else:
            raise IOError('Wrong mode, check please.')
        # After plot, annotate cell name.
        base_graph = gt.Graph_Depth_Change(base_graph,'u1')
        from PIL import ImageFont
        from PIL import Image
        from PIL import ImageDraw
        font = ImageFont.truetype('arial.ttf',15)
        im = Image.fromarray(base_graph)
        y,x = c_cell_info.centroid
        draw = ImageDraw.Draw(im)
        draw.text((x+10,y+10),cell_name,(0,255,100),font = font,align = 'center')
        annotated_graph = np.array(im)
        #base_graph = gt.Clip_And_Normalize(base_graph,5)
        gt.Show_Graph(annotated_graph, cell_name+'_Annotate', cell_save_path,show_time = show_time)
        return True
    
    def Part_Cell_Plotter(self,cell_used,mode = 'circle'):
        
        '''
        Plot part of cells.
        
        Parameters
        ----------
        cell_used : (list)
            List of cell name you want to plot.
        mode : 'fill' or 'circle', optional
            Mode of cell annotation. The default is 'circle'.

        '''
        
        base_graph = self.average_graph
        if mode == 'fill':
            base_graph = cv2.cvtColor(base_graph,cv2.COLOR_GRAY2RGB)*0.7
        elif mode == 'circle':
            base_graph = cv2.cvtColor(base_graph,cv2.COLOR_GRAY2RGB)
        used_cell_num = len(cell_used)
        for i in range(used_cell_num):
            c_cell_name = cell_used[i]
            c_cell_dic = self.all_cell_dic[c_cell_name]
            c_cell_info = c_cell_dic['Cell_Info']
            if mode == 'fill':
                cell_y,cell_x = c_cell_info.coords[:,0],c_cell_info.coords[:,1]
                base_graph[cell_y,cell_x,1] +=32768
            elif mode == 'circle':
                loc_y,loc_x = c_cell_info.centroid
                base_graph = cv2.circle(base_graph,(int(loc_x),int(loc_y)),radius = 10,color = (0,0,65535),thickness=2)
        base_graph = np.clip(base_graph,0,65535).astype('u2')
        gt.Show_Graph(base_graph, 'Part_Annotated_Graph', self.save_folder)
    
    def Part_Cell_F_Disp(self,cell_name_list,graph_name = 'Part_Cell',bins = 10):
        all_F_value = []
        for i in range(len(cell_name_list)):
            all_F_value.append(self.all_cell_dic[cell_name_list[i]]['Average_F'])
        fig, ax = plt.subplots(figsize = (6,4))
        ax.set_title(graph_name)
        hc_mean = np.array(all_F_value).mean()
        hc_std = np.array(all_F_value).std()
        ax.hist(all_F_value,bins = bins)
        ax.annotate('Average:'+str(round(hc_mean,2)),xycoords='figure fraction',xy=(0.7, 0.83))
        ax.annotate('STD:'+str(round(hc_std,2)),xycoords='figure fraction',xy=(0.7, 0.78))
        fig.savefig(self.save_folder+'\\'+graph_name,dpi = 120)
    
    
    def Index_Calculator_Core(self,run_name,A_ID_list,B_ID_list,
                              used_frame = [4,5],mode = 'processed'):
        '''
        Generate single pair tuning index.

        Parameters
        ----------
        run_name : (str)
            Single run name of calculation.
        A_ID_list : (list)
            List of A IDs.
        B_ID_list : (list)
            List of B IDs.
        used_frame : (list), optional
            Which frames we use in a single condition. The default is [4,5].
        mode : 'processed' or 'raw', optional
            Which CR Train to use. The default is 'processed'.

        Returns
        -------
        all_Index : (Dic)
            Index of all cell tuning informations.

        '''
        all_Index = {}
        for i in range(self.cell_num):
            A_sets = []
            B_sets = []
            c_cell_name = self.all_cell_names[i]
            if self.all_cell_dic[c_cell_name]['In_Run'][run_name] == False: # pass if not in run
                all_Index[c_cell_name] = None    
                continue
            else:
                all_Index[c_cell_name] = {}
            if mode == 'processed':
                c_series = self.all_cell_dic[c_cell_name][run_name]['CR_Train']
            elif mode == 'raw':
                c_series = self.all_cell_dic[c_cell_name][run_name]['Raw_CR_Train']
            for j in range(len(A_ID_list)):
                c_part = c_series[A_ID_list[j]][:,used_frame]
                A_sets.extend(list(c_part.flatten()))
            for j in range(len(B_ID_list)):
                c_part = c_series[B_ID_list[j]][:,used_frame]
                B_sets.extend(list(c_part.flatten()))
            sample_size = min(len(A_sets),len(B_sets))
            A_for_index = np.array(random.sample(A_sets,sample_size))
            B_for_index = np.array(random.sample(B_sets,sample_size))
            A_mean = A_for_index.mean()
            B_mean = B_for_index.mean()
            tuning_index = (A_mean-B_mean)/(A_mean+B_mean)
            all_Index[c_cell_name]['Tuning_Index'] = tuning_index
            # Do t test next.
            from scipy.stats import ttest_rel
            t_value,p_value = ttest_rel(A_for_index,B_for_index)
            all_Index[c_cell_name]['t_value'] = t_value
            all_Index[c_cell_name]['p_value'] = p_value
            all_Index[c_cell_name]['Origin_Value'] = (A_mean,B_mean)
            all_Index[c_cell_name]['Cohen_D'] = t_value/np.sqrt(sample_size)
        return all_Index
    
    def Black_Cell_Identifier(self,run_name_lists,used_frame = [4,5],p_thres = 0.01,mode ='processed'):
        '''
        Find black cells in given run lists.

        Parameters
        ----------
        run_name_lists : (list)
            List of stim runs you want to use.
        used_frame : (list), optional
            Which frames are used as stim on. ROI can be different! The default is [4,5].
        p_thres : (float), optional
            P value thres of regard significant. The default is 0.01.
        mode : ('processed' or 'raw'), optional
            Based on CR or Raw CR. The default is 'processed'.

        Returns
        -------
        Black_Cell_Information : (Dic)
            Dictionary of all black cells. Non black cell not included.

        '''
        Black_Cell_Information = {}
        for i in range(self.cell_num):# cycle_all_cell
            c_cell = self.all_cell_dic[self.all_cell_names[i]]
            c_black_dic = {}# for recording all black infos
            for j in range(len(run_name_lists)):# cycle all run
                c_run = run_name_lists[j]
                c_black_dic[c_run] = {}
                if c_cell['In_Run'][c_run] == True:# process if only cell in graph.
                    if mode == 'processed':
                        CR = c_cell[c_run]['CR_Train']
                    elif mode == 'raw':
                        CR = c_cell[c_run]['Raw_CR_Train']
                    else:
                        raise IOError('Wrong mode, please check.')
                else:
                    del c_black_dic[c_run]
                    continue
                # get CR dics, then compare each cond with 0.
                all_cons = list(CR.keys())
                if 0 in all_cons:# If we have 0
                    all_cons.remove(0)
                    blank_set = CR[0][:,used_frame].flatten()
                    # single cond calculation.
                    for k in range(len(all_cons)):
                        c_con = all_cons[k]
                        sc_set = CR[c_con][:,used_frame].flatten()
                        c_t,c_p,c_d = st.T_Test_Pair(sc_set, blank_set)
                        before_set = CR[c_con][:,[0,1]].flatten()
                        before_t,before_p,_ = st.T_Test_Pair(sc_set, before_set)
                        if ((c_t<0 and c_p<p_thres) and (before_t<0 and before_p<p_thres)):
                            c_black_dic[c_run][c_con] = (c_t,c_p,c_d)
                else:# if this run have no 0
                    for k in range(len(all_cons)):
                        c_con = all_cons[k]
                        blank_set = CR[c_con][:,[0,1]].flatten()
                        sc_set = CR[c_con][:,used_frame].flatten()
                        c_t,c_p,c_d = st.T_Test_Pair(sc_set, blank_set)
                        if (c_t<0 and c_p<p_thres):
                            c_black_dic[c_run][c_con] = (c_t,c_p,c_d)
                # Then delete empty run.
                if c_black_dic[c_run] == {}:
                    del c_black_dic[c_run]
            if c_black_dic != {}:
                Black_Cell_Information[c_cell['Name']] = c_black_dic
                    
                    
        return Black_Cell_Information
    
    def T_Map_Plot_Core(self,run_name,A_ID_list,B_ID_list,p_thres = 0.05,plot = True,
                              used_frame = [4,5],mode = 'processed'):
        current_index_dic = self.Index_Calculator_Core(run_name, A_ID_list, B_ID_list)
        all_t = {}
        for i in range(len(self.all_cell_names)):
            if current_index_dic[self.all_cell_names[i]] != None:
                c_t = current_index_dic[self.all_cell_names[i]]['t_value']
                if current_index_dic[self.all_cell_names[i]]['p_value']<p_thres:
                    all_t[self.all_cell_names[i]]=c_t
        t_data = np.zeros(self.average_graph.shape,dtype = 'f8')
        plotted_cells = list(all_t.keys())
        for i in range(len(plotted_cells)):
            c_cell_info = self.all_cell_dic[plotted_cells[i]]['Cell_Info']
            cell_y,cell_x = c_cell_info.coords[:,0],c_cell_info.coords[:,1]
            t_data[cell_y,cell_x] = all_t[plotted_cells[i]]
        if plot == True:
            fig = plt.figure(figsize = (15,15))
            plt.title('T_Map',fontsize=36)
            fig = sns.heatmap(t_data,square=True,yticklabels=False,xticklabels=False,center = 0)
            fig.figure.savefig(self.save_folder+r'\T_Map.png')
            plt.clf()
            norm_t_data = t_data/abs(t_data).max()
            posi_part = norm_t_data*(norm_t_data>0)*65535
            nega_part = norm_t_data*(norm_t_data<0)*65535
            folded_map = cv2.cvtColor(self.average_graph,cv2.COLOR_GRAY2RGB)*0.7
            folded_map[:,:,0] -= nega_part
            folded_map[:,:,2] += posi_part
            folded_map = np.clip(folded_map,0,65535).astype('u2')
            cv2.imwrite(self.save_folder+r'\T_Map_Folded.png',folded_map)
        return t_data
        
        