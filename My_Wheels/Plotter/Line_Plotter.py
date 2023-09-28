# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 16:43:00 2021

@author: ZR
"""
import seaborn as sns
import matplotlib.pyplot as plt


def EZLine(y_series, x_series = 'Default',
           save_folder = 'None',graph_name = 'Line',dpi = 180,
           title = 'Line', figsize = (12,8),
           x_label = 'X',y_label = 'Y',
           x_range = 'Default',y_range = 'Default'
           ):
    '''
    Easy used single line plotter, can go with single line data, and have limited parameters 

    Parameters
    ----------
    y_series : (list or nd array)
        Data you want to plot.
    x_series : (x value of y data), optional
        X location lists. The default is 'Default'.
    save_folder : (str), optional
        Path to save graph. The default is 'None'.
    graph_name : (str), optional
        Name of graph to save. The default is 'Line'.
    dpi : (int), optional
        Dpi of output graph. The default is 180.
    title : (str), optional
        Graph title. The default is 'Line'.
    figsize : (2-element-turple), optional
        Size of graph we plot(in inches). The default is (12,8).
    x_label : (str), optional
        Label of X axis. The default is 'X'.
    y_label : (str), optional
        Label of Y axis. The default is 'Y'.
    x_range : (2-element-list/turple), optional
        Range of X series. The default is 'Default'.
    y_range : (2-element-list/turple), optional
        Range of Y series. The default is 'Default'.

    Returns
    -------
    bool
        DESCRIPTION.

    '''
    
    # Plot parts
    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(1,1,1)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    if x_series == 'Default':
        x_series = range(len(y_series))
    sns.lineplot(x = x_series,y = y_series)
    ax.set_title(title,fontsize = figsize[0]*2)
    ax.set_xlabel(x_label,fontsize = figsize[1]*2)
    ax.set_ylabel(y_label,fontsize = figsize[1]*2)
    if x_range != 'Default':
        ax.set_xlim(x_range)
    if y_range != 'Default':
        ax.set_ylim(y_range)
    # Save part
    if save_folder != 'None':
        fig.savefig(save_folder+r'\\'+graph_name+'.png',dpi = dpi)
    return True

def Multi_Line_Plot(line_lists,legends, x_series = 'Default',
                    save_folder = 'None',graph_name = 'Line',dpi = 180,
                    title = 'Line', figsize = (12,8),
                    x_label = 'X',y_label = 'Y',
                    x_range = 'Default',y_range = 'Default'
                    ):
    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(1,1,1)
    if x_series == 'Default':
        x_series = range(len(line_lists[0]))
    for i,c_line in enumerate(line_lists):
        sns.lineplot(x = x_series,y = line_lists[i],label = legends[i])
    ax.legend(prop={'size': 20})
    ax.set_title(title,fontsize = figsize[0]*2)
    ax.set_xlabel(x_label,fontsize = figsize[1]*2)
    ax.set_ylabel(y_label,fontsize = figsize[1]*2)
    if x_range != 'Default':
        ax.set_xlim(x_range)
    if y_range != 'Default':
        ax.set_ylim(y_range)
    # Save part
    if save_folder != 'None':
        fig.savefig(save_folder+r'\\'+graph_name+'.png',dpi = dpi)
    return True







def Plot_with_Given_Ticks():
    
    pass
