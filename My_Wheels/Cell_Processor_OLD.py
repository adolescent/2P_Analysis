# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 15:21:30 2021

@author: ZR

Define a class to include any data processing method based on cell data.
Use class so new method can be included easily.

"""
import  Stim_Dic_Tools as SDT
import numpy as np
import OS_Tools_Kit as ot
import matplotlib.pyplot as plt
import My_Wheels.Filters as Filters


class Cell_Processor(object):
    
    def __init__(self,day_folder,
                 runname,
                 average_graph = None,
                 series_mode = 'F',
                 filter_para = (0.02,False)
                 
                 ):
        '''
        Some basic information for calculation
        
        Parameters
        ----------
        day_folder : (str)
            Day folder of runs. All variables in this folder.
        runname : (str)
            Run name. Format as 'Run001'.
        average_graph : (2D Array), optional
            Average graph in a day. For cell annotate. The default is None.
        series_mode : ('F' or 'dF'), optional
            Which series to use, raw F series or dF series. F is recommended. The default is 'F'.
        filter_para : (2 element truple), optional
            HP and LP filter para. Detail in Filters. The default is (0.02,False),~0.013Hz HP.

        '''
        # Read in variables
        print('Make sure cell data and SFA data in day folder.')
        cell_data_path = ot.Get_File_Name(day_folder,'.ac')[0]
        all_stim_dic_path = ot.Get_File_Name(day_folder,'.sfa')[0]
        self.all_cell_dic = ot.Load_Variable(cell_data_path)
        self.average_graph = average_graph
        self.all_stim_dic = ot.Load_Variable(all_stim_dic_path)
        self.stim_frame_align = self.all_stim_dic[runname]
        # Then get each cell spike train.
        cell_num = len(self.all_cell_dic)
        self.all_cells_train = {}
        self.all_cell_names = list(self.all_cell_dic.keys())
        for i in range(cell_num):
            current_name = self.all_cell_names[i]
            if series_mode == 'F':
                cell_series = self.all_cell_dic[current_name]['F_train'][runname]
            elif series_mode == 'dF':
                cell_series = self.all_cell_dic[current_name]['dF_F_train'][runname]
            else:
                raise IOError('Invalid input mode, please check.')
            cell_series = Filters.Signal_Filter(cell_series,filter_para = filter_para)# Then filter cell train
            self.all_cells_train[current_name] = cell_series



    def Single_Cell_Response_Data(self,Condition_dics,
                     cell_name,
                     head_frame = 3,tail_frame = 3,
                     condition_dF_base = [0,1,2]
                     ):
        '''
        Generate single cell response data. This is the initial step of most functions.

        Parameters
        ----------
        Condition_dics : (Dic)
            Condition name and stim id lists. Generate by Stim_ID_Combiner.
        cell_name : (str)
            name of current cell.
        head_frame : (int), optional
            How many frame before stim onset. The default is 3.
        tail_frame : (int), optional
            How many frame after stim onset. The default is 3.
        condition_dF_base : list, optional
            Which frame will be used as dR base. The default is [0,1,2].

        Returns
        -------
        self.raw_response_data : (dic)
            Raw response F data. 
        self.response_data : (dic)
            adjusted response F data.
        self.current_cell : (str)
            current cell name.

        '''
        # Initialization
        subgraph_num = len(Condition_dics)
        all_subgraph_name = list(Condition_dics.keys())
        cell_train = self.all_cells_train[cell_name]
        self.raw_response_data = {}
        self.response_data = {}
        self.current_cell = cell_name
        # For every map.
        for i in range(subgraph_num):
            current_subgraph_name = all_subgraph_name[i]
            current_stim_id = Condition_dics[current_subgraph_name]
            all_conditions = SDT.Frame_ID_Extrator_In_Conditions(self.stim_frame_align, 
                                                                 current_stim_id,
                                                                 head_extend= head_frame,
                                                                 tail_extend=tail_frame)
            condition_num = len(all_conditions)
            condition_length = len(all_conditions[0])
            self.raw_response_data[all_subgraph_name[i]] = np.zeros(shape = (condition_num,condition_length),dtype = 'f8')
            self.response_data[all_subgraph_name[i]] = np.zeros(shape = (condition_num,condition_length),dtype = 'f8')       
            
            # Fill in response data matrix.
            for j in range(condition_num):
                current_frame_id = all_conditions[j]
                current_train = cell_train[current_frame_id]
                self.raw_response_data[all_subgraph_name[i]][j,:] = current_train
                if condition_dF_base != None:
                    current_base = current_train[condition_dF_base].mean()
                    dR_train = current_train/current_base-1
                    self.response_data[all_subgraph_name[i]][j,:] = dR_train
        if condition_dF_base == None:
            print('No dF/F calculation, origin train returned.')
            self.response_data = None  
        return self.raw_response_data,self.response_data,self.current_cell
    
    def Average_Response_Map(self,data_mode = 'processed',
                             stim_on = (3,6),
                             error_bar = True):
        '''
        Generate Average Response map.

        Parameters
        ----------
        data_mode : 'processed' or 'raw', optional
            Which response data to use.raw will use F value, processed use dF. The default is 'processed'.
        stim_on : (turple), optional
            Frame range of stim on. The default is (3,6).
        error_bar : bool, optional
            Whether we annotat error bar on graphs. The default is True.

        Returns
        -------
        fig : (plt fig file)
            Current cell response frame. Remenber to plt.clf() after save!.

        '''
        # calculate average response data first.
        cell_name = self.current_cell
        if data_mode == 'processed':
            plotable_data = self.response_data
        elif data_mode == 'raw':
            plotable_data = self.raw_response_data
        else:
            raise IOError('Data mode invalid.')
        subgraph_num = len(plotable_data)
        all_subgraph_name = list(plotable_data.keys())
        self.response_plot_dic = {}
        # get y sticks
        y_max = 0
        y_min = 65535
        for i in range(subgraph_num):
            current_graph_response = plotable_data[all_subgraph_name[i]]
            average_plot = current_graph_response.mean(0)
            average_std = current_graph_response.std(0)
            self.response_plot_dic[all_subgraph_name[i]] = (average_plot,average_std)
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
        fig.suptitle(cell_name+'_Response Maps', fontsize=30)
        for i in range(subgraph_num):
            current_col = i%col_num
            current_row = i//col_num
            current_graph_name = all_subgraph_name[i]
            current_data = self.response_plot_dic[current_graph_name]
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
        return fig
    
    def Single_Condition_Response_Map(self,data_mode = 'raw'):
        pass


    def Radar_Map(self,radar_dic,
                  mode = 'processed',
                  error_bar = True,
                  base_range = [0,1,2],
                  stim_on_range = [3,4,5,6]):
        
        cell_name = self.current_cell
        if data_mode == 'processed':
            plotable_data = self.response_data
        elif data_mode == 'raw':
            plotable_data = self.raw_response_data
        else:
            raise IOError('Data mode invalid.')
        # Then 
        
        
    
    
    
    def Wavelet_Specs(self):
        # This is an example. 
        import pywt
        import numpy as np
        import matplotlib.pyplot as plt
        x = np.arange(512)
        y = np.sin(2*np.pi*x/32)
        coef, freqs=pywt.cwt(y,np.arange(1,512),'cgau7')
        plt.matshow(coef) # doctest: +SKIP
        plt.show()
    def Cross_Wavelet(self):
        # Cross power specturn calculator
        pass
        
    
        
        
        

    
