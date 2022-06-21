# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 16:34:49 2021

@author: ZR
"""

import  Stim_Dic_Tools as SDT
import numpy as np
import OS_Tools_Kit as ot
import matplotlib.pyplot as plt

class Cell_Processor(object):
    
    def __init__(self,day_folder,average_graph = None):
        print('Make sure cell data and SFA data in day folder.')
        cell_data_path = ot.Get_File_Name(day_folder,'.ac')[0]
        all_stim_dic_path = ot.Get_File_Name(day_folder,'.sfa')[0]
        self.all_cell_dic = ot.Load_Variable(cell_data_path)
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
        graph_folder = self.save_folder+r'\\'+runname
        ot.mkdir(graph_folder)
        for i in range(self.cell_num):# all cells
            c_cellname = self.all_cell_names[i]
            # get cr trains    
            if mode == 'processed':
                if runname in self.all_cell_dic[c_cellname]['CR_trains']:
                    cr_train = self.all_cell_dic[c_cellname]['CR_trains'][runname]
                else:
                    cr_train = None
            elif mode == 'raw':
                if runname in self.all_cell_dic[c_cellname]['Raw_CR_trains']:
                    cr_train = self.all_cell_dic[c_cellname]['Raw_CR_trains'][runname]
                else:
                    cr_train = None
                    
            # generate plotable data.
            if cr_train == None:# no cr train, continue check another cell.
                continue
            else:#Combine conditions to get plotable data
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
                average_std = current_graph_response.std(0)
                response_plot_dic[all_subgraph_name[j]] = (average_plot,average_std)
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
    
    
    
    def Radar_Maps(self,runname,Radar_Cond,
                   on_frames = [3,4,5,6],bais_angle = 0,mode = 'processed',error_bar = True):
        radar_folder = self.save_folder+r'\\'+runname+'_Radar_Maps'
        ot.mkdir(radar_folder)
        # Cycle all cells
        for i in range(self.cell_num):
            # get cr train
            c_cellname = self.all_cell_names[i]
            if mode == 'processed':
                if runname in self.all_cell_dic[c_cellname]['CR_trains']:
                    cr_train = self.all_cell_dic[c_cellname]['CR_trains'][runname]
                else:
                    cr_train = None
            elif mode == 'raw':
                if runname in self.all_cell_dic[c_cellname]['Raw_CR_trains']:
                    cr_train = self.all_cell_dic[c_cellname]['Raw_CR_trains'][runname]
                else:
                    cr_train = None
            if cr_train == None:
                continue
            # get radar data and std
            radar_data = SDT.CR_Train_Combiner(cr_train,Radar_Cond)
            all_radar_names = list(radar_data.keys())
            plotable_data = {}
            plotable_data['Names'] = []
            plotable_data['Values'] = np.zeros(len(all_radar_names),dtype = 'f8')
            plotable_data['Stds'] = np.zeros(len(all_radar_names),dtype = 'f8')
            for j in range(len(all_radar_names)):
                c_name = all_radar_names[j]
                plotable_data['Names'].append(c_name)
                c_conds,c_stds = radar_data[c_name].mean(0),radar_data[c_name].std(0)
                cutted_conds,cutted_std = c_conds[on_frames],c_stds[on_frames]
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
    def Get_Trains(self,runname,train_name):
        '''
        Get dictionary of single train single mode.

        Parameters
        ----------
        runname : (str)
            Runname .
        train_name : ('F_train' or 'dF_F_train')
            Train method. If not properly given, an empty dic will be given.

        Returns
        -------
        cell_trains : (Dic)
            All cell having specific train will be shown.

        '''
        print('Get single run specific trains.')
        cell_trains = {}
        for i in range(self.cell_num):
            c_cell_name = self.all_cell_names[i]
            if runname in self.all_cell_dic[c_cell_name][train_name]:
                cell_trains[c_cell_name] = self.all_cell_dic[c_cell_name][train_name][runname]
        return cell_trains
    
    
    
    