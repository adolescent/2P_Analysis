# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 14:55:18 2021

@author: ZR
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolor


def EZHist(data,label = 'Distribution',bins = 'auto',
           save_folder = 'None',graph_name = 'Density',dpi = 180,
           title = 'Density',figsize = (6,8),
           x_label = 'X',y_label = 'Density',
           x_range = 'Default'
           ):
    '''
    Plot simple histograph.

    Parameters
    ----------
    data : (nd array,pandas frame or list)
        Data you want to do histo.
    label : (str), optional
        Label of data. The default is 'Distribution'.
    bins : (int or 'auto'), optional
        How many bings in your histogram. Set auto can auto select. The default is 'auto'.
    save_folder : (str), optional
        Folder to save graphs. The default is 'None'.
    graph_name : (str), optional
        Graph file name. The default is 'Density'.
    dpi : (int), optional
        Graph dpi. The default is 180.
    title : (str), optional
        Title of graph. The default is 'Density'.
    figsize : (2-element-turple), optional
        Figure size in inches. The default is (6,8).
    x_label : (str), optional
        Label of X axis. The default is 'X'.
    y_label : (str), optional
        Label of Y axis. The default is 'Density'.
    x_range : (2-element-turple), optional
        Range of X axis. Use default to skip. The default is 'Default'.

    Returns
    -------
    None.

    '''
    used_data = np.array(data).flatten()
    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(1,1,1)
    sns.histplot(used_data,bins =bins,label = label,kde = True,stat = 'density')
    ax.legend(prop={'size': 15})
    ax.set_title(title,fontsize = figsize[1]*2)
    ax.set_xlabel(x_label,fontsize = figsize[1]*2)
    ax.set_ylabel(y_label,fontsize = figsize[1]*2)
    if x_range != 'Default':
        ax.set_xlim(x_range)
    # Save part
    if save_folder != 'None':
        fig.savefig(save_folder+r'\\'+graph_name+'.png',dpi = dpi)
        
        
        
        
        
def Multi_Hist_Plot(data_lists,label_lists,bins = 'auto',
                    save_folder = 'None',graph_name = 'Density',dpi = 180,
                    title = 'Density',figsize = (6,8),
                    x_label = 'X',y_label = 'Density',
                    x_range = 'Default'
                    ):
    '''
    Draw many distribution in single graph

    Parameters
    ----------
    data_lists : (list)
        List of data sets.
    label_lists : (list)
        List of data labels. Need to be corresponed to data.
    bins : (int), optional
        Bins of histograph. Can be skipeed by auto. The default is 'auto'.
    save_folder : (str), optional
        Graph save folder. The default is 'None'.
    graph_name : (str), optional
        Graph file name. The default is 'Density'.
    dpi : (int), optional
        Graph dpi. The default is 180.
    title : (str), optional
        Graph title. The default is 'Density'.
    figsize : (2-element-turple), optional
        Graph size in inches. The default is (6,8).
    x_label : (str), optional
        Label of X axis. The default is 'X'.
    y_label : (str), optional
        Label of Y axis. The default is 'Density'.
    x_range : (2-element-turple), optional
        Range of X axis. The default is 'Default'.

    Returns
    -------
    None.

    '''
    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(1,1,1)
    colorbar = list(mcolor.TABLEAU_COLORS.keys())
    for i,c_data in enumerate(data_lists):
        used_data = np.array(c_data).flatten()
        sns.histplot(used_data,bins =bins,label = label_lists[i],kde = True,stat = 'density',color = colorbar[i],alpha = 0.7)
    ax.legend(prop={'size': 15})
    ax.set_title(title,fontsize = figsize[1]*2)
    ax.set_xlabel(x_label,fontsize = figsize[1]*2)
    ax.set_ylabel(y_label,fontsize = figsize[1]*2)
    if x_range != 'Default':
        ax.set_xlim(x_range)
    # Save part
    if save_folder != 'None':
        fig.savefig(save_folder+r'\\'+graph_name+'.png',dpi = dpi)
        
    
    
    
    
