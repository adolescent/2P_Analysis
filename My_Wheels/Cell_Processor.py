# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 16:34:49 2021

@author: ZR
"""

import  Stim_Dic_Tools as SDT
import numpy as np
import OS_Tools_Kit as ot
import matplotlib.pyplot as plt
import My_Wheels.Filters as Filters

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
                           error_bar = True
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
            col_num = int(np.ceil(np.sqrt(subgraph_num)))
            row_num = int(np.ceil(subgraph_num/col_num))
            fig,ax = plt.subplots(row_num,col_num,figsize = (15,15))# Initialize graphs
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
            ot.Save_Variable(graph_folder, c_cellname+'_Response_Data', response_plot_dic)
            fig.savefig(graph_folder+r'\\'+c_cellname+'_Response.png',dpi = 180)
            plt.clf()
            plt.close()
        return True
    def Radar_Maps(self,runnane,Radar_Conds,
                   on_frames = [3,4,5,6],mode = 'processed'):
        
        return True