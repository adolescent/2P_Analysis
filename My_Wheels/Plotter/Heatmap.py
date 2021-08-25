# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 14:26:53 2021

@author: ZR
"""
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def Heat_Maps(input_frame,
              square = False,center = None,
              xticklabels = 'auto',yticklabels = 'auto',
              save_folder = 'None',graph_name = 'Heat_Map',dpi = 180,
              title = 'Heat Map', figsize = (20,12),
              x_label = 'Time',y_label = 'Value',
              y_tick_num = 20,x_tick_num = 20,
              ):
    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(1,1,1)
    sns.heatmap(input_frame,square = square,center = center,xticklabels=xticklabels,yticklabels=yticklabels)
    ax.set_title(title,fontsize = figsize[0]*2)
    ax.set_xlabel(x_label,fontsize = figsize[1]*2)
    ax.set_ylabel(y_label,fontsize = figsize[1]*2)
    # get y ticks label
    if yticklabels != False:
        all_y_ticks = input_frame.index.tolist()
        yticks = np.linspace(0,len(all_y_ticks)-1,y_tick_num,dtype = np.int)
        yticklabels = [all_y_ticks[idx] for idx in yticks]
        ax.set_yticks(yticks)
        ax.set_yticklabels(np.round(yticklabels,2),rotation = 20,fontsize = 18)
    # get x ticks label
    if xticklabels != False:
        all_x_ticks = input_frame.columns.tolist()
        xticks = np.linspace(0,len(all_x_ticks)-1,x_tick_num,dtype = np.int)
        xticklabels = [all_x_ticks[idx] for idx in xticks]
        ax.set_xticks(xticks)
        ax.set_xticklabels(np.round(xticklabels,2),rotation = 20,fontsize = 18)
    
    
    if save_folder != 'None':
        fig.savefig(save_folder+r'\\'+graph_name+'.png',dpi = dpi)
    return True



    