
'''
This script include multiple vital functions for cell tuning calculation.
Codes following will generate:
1. Single cell tuning property(t values of OD/Orien/Color etc..)
2. T test maps (in format cell t value matrix)
3. Cell Response Curve. avr/std of each ID response.(for all cells)
4. This tool need to be updatable.
'''
#%%
from My_Wheels.Cell_Class.Format_Cell import Cell
import My_Wheels.Stim_Dic_Tools as SDT
import numpy as np
from Standard_Parameters.Stim_Name_Tools import Stim_ID_Combiner
import Cell_Class.Cell_Class_Tools as Class_Tools
import OS_Tools_Kit as ot
import matplotlib.pyplot as plt
from Standard_Parameters.Sub_Graph_Dics import Sub_Dic_Generator
from Decorators import Timer
import pandas as pd


class Stim_Cells(Cell):
    
    def __init__(self,*args, **kwargs): # Success all functions & variables in parent class Cell.
        super().__init__(*args, **kwargs)
        # And copy all variables.
        self.Calculate_All_CR_Response()
        self.Calculate_All_Stim_Response()

    def Get_CR_Response_Core(self,runname='1-006',head_extend=3,tail_extend = 3):
        # Input a frame, generate CR response of all cells in given frame.
        style2_runname = 'Run'+runname[2:]
        c_stim_frame_align = self.Stim_Frame_Align[style2_runname]
        self.Cell_Response = {}
        condition_frames = SDT.Condition_Response_Frames(c_stim_frame_align,head_extend,tail_extend)
        # get each condition have same length.
        condition_length = 65535
        all_conditions = list(condition_frames.keys())
        for i in range(len(all_conditions)):# get proper length. Use least length.
            current_cond_length = len(condition_frames[all_conditions[i]][0])
            if current_cond_length < condition_length:
                condition_length = current_cond_length
        for i in range(len(all_conditions)):# cut too long condition.
            current_condition = condition_frames[all_conditions[i]] # This is shallow copy so change below will be returned too.
            for j in range(len(current_condition)):
                current_condition[j] = current_condition[j][:condition_length]

        # Calculate Condition Response.
        c_Z_frame = self.Z_Frames[runname]
        all_cell_cr_dic = {}
        for j,cc in enumerate(self.acn):
            c_Z_train = np.array(c_Z_frame[cc])
            all_cell_cr_dic[cc] = {}
            for i,c_condition in enumerate(all_conditions):
                c_frame_lists = condition_frames[c_condition]
                Z_matrix = np.zeros(shape = (len(c_frame_lists[0]),len(c_frame_lists)),dtype = 'f8')
                for j in range(len(c_frame_lists)):
                    cs_cond = c_frame_lists[j]
                    Z_matrix[:,j] = c_Z_train[cs_cond]
                all_cell_cr_dic[cc][c_condition] = Z_matrix
        return all_cell_cr_dic
    
    def Calculate_All_CR_Response(self):
        if self.od_type != False:
            self.od_CR_Response = self.Get_CR_Response_Core(runname = self.odrun)
        if self.orien_type != False:
            self.orien_CR_Response = self.Get_CR_Response_Core(runname = self.orienrun)
        if self.color_type != False:
            self.color_CR_Response = self.Get_CR_Response_Core(runname = self.colorrun)
    
    def Calculate_All_Stim_Response(self):
        # Must be done after All CR_Response Calculation finish.
        self.Stim_Reponse_Dics = {}
        if self.od_type == 'OD_2P':
            self.Stim_Reponse_Dics['OD'] = Class_Tools.All_Cell_Condition_Combiner(self.od_CR_Response,Condition_Dics=Stim_ID_Combiner('OD_2P'))
        if self.orien_type == 'G16':# orientation will return both dir and orientation methods.
            self.Stim_Reponse_Dics['Oriens'] = Class_Tools.All_Cell_Condition_Combiner(self.orien_CR_Response,Condition_Dics=Stim_ID_Combiner('G16_Oriens'))
            self.Stim_Reponse_Dics['Dirs'] = Class_Tools.All_Cell_Condition_Combiner(self.orien_CR_Response,Condition_Dics=Stim_ID_Combiner('G16_Dirs'))
        if self.color_type == 'Hue7Orien4':
            self.Stim_Reponse_Dics['Colors'] = Class_Tools.All_Cell_Condition_Combiner(self.color_CR_Response,Condition_Dics=Stim_ID_Combiner('Hue7Orien4_Colors'))
            
    def Plot_Stim_Response(self,stim = 'OD',stim_on = (3,6),error_bar = True,figsize =(15,15)):
        # This function can plot stim response curve. Add multiple visualization.
        foldername = stim+'_Response_Curves'
        c_savepath = ot.join(self.wp,foldername)
        ot.mkdir(c_savepath)
        c_stimdata = self.Stim_Reponse_Dics[stim] # get all cell stim data.
        if stim == 'OD':
            graph_shape = (3,5)
        elif stim == 'Oriens':
            graph_shape = (3,3)
        elif stim == 'Dirs':
            graph_shape = (3,8)
        elif stim == 'Colors':
            graph_shape = (2,4)
        for i,cc in enumerate(self.acn):
            cc_stim_data = c_stimdata[cc]
            subgraph_num = len(cc_stim_data)
            all_subgraph_name = list(cc_stim_data.keys())
            y_max = 0# get y sticks
            y_min = 65535
            response_plot_dic = {}
            for j,c_subgraph in enumerate(all_subgraph_name):
                current_graph_response = cc_stim_data[c_subgraph]
                average_plot = current_graph_response.mean(1)
                se_2 = current_graph_response.std(1)/np.sqrt(current_graph_response.shape[1])*2
                response_plot_dic[c_subgraph] = (average_plot,se_2)
                # renew y min and y max.
                if average_plot.min() < y_min:
                    y_min = average_plot.min()
                if average_plot.max() > y_max:
                    y_max = average_plot.max()
            y_range = [y_min-0.3,y_max+0.3]
            # then we will plot..
            fig,ax = plt.subplots(graph_shape[0],graph_shape[1],figsize = figsize)
            fig.suptitle('Cell'+str(cc)+'_'+foldername, fontsize=30)
            for j in range(subgraph_num):
                current_col = j%graph_shape[1]
                current_row = j//graph_shape[1]
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
            fig.savefig(c_savepath+r'\\'+str(cc)+'_Response.png',dpi = 180)
            plt.clf()
            plt.close()
        
    @Timer
    def Plot_All_Tuning_Curve(self):
    # This is just a full version of all 4 stim response plot.
        print('This will cost some time, just be patient.')
        all_stim_types = list(self.Stim_Reponse_Dics.keys())
        for i,c_stim in enumerate(all_stim_types):
            self.Plot_Stim_Response(stim=c_stim)

    ############### Functions above are CR responses, Below are Tunings #############
    def T_Calculator_Core(self,cr_response,A_set,B_set,p_thres = 0.05,used_frame = [4,5]):
        # This function will generate single t test. Input all cell CR Resposne and Generated Subdic
        # Cycle all graphs.
        ttest_frame = pd.DataFrame(index = self.acn,columns = ['t_value','p_value','A_reponse','B_response',])
        
        return ttest_frame
        
            
            
        
        
    def Calculate_Cell_Tunings(self):
        # Calculate all cell tunings, return a cell data frame of tuning t/p.
        pass
    def Get_Labels(self):
        # get label of given frame, can be OD,OD-orien,orien,dir,color etc.
        pass
    
    
    
#%%
if __name__ == '__main__':
    day_folder = r'E:\2P_Raws\220630_L76_2P'
    test_cell = Stim_Cells(day_folder,od = 6,orien = 7, color = 8)
    # test_cell.Get_All_Stim_Response()
    # test_cell.Plot_Stim_Response(stim = 'OD')
    