'''
Provide multiple plot tools to save code lines.
'''

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from mpl_toolkits import mplot3d
from matplotlib.animation import FuncAnimation
from Cell_Tools.Cell_Visualization import Cell_Weight_Visualization


def Plot_3D_With_Labels(data,labels,ncol = 2):
    n_clusters = len(set(labels))
    colors = cm.turbo(np.linspace(0, 1, n_clusters+1))# colorbars.
    fig = plt.figure(figsize = (12,10))
    ax = plt.axes(projection='3d')
    ax.grid(False)
    unique_labels = np.unique(labels)
    handles = []
    all_scatters = []
    counter = 0
    for label in unique_labels:
        mask = labels == label
        scatter = ax.scatter3D(data[:,0][mask], data[:,1][mask], data[:,2][mask], label=label,s = 5,color = colors[counter])
        all_scatters.append(scatter)
        handles.append(scatter)
        counter +=1
    ax.legend(handles=handles,ncol = 2)
    
    return fig,ax

def Save_3D_Gif(ax,fig):
    def update(frame):
        ax.view_init(elev=25, azim=frame)  # Update the view angle for each frame
        return ax,
    animation = FuncAnimation(fig, update, frames=range(0, 360, 5), interval=150)
    animation.save('3D_plot.gif', writer='pillow')
    return True

def Plot_Multi_Subgraphs(graph_frame,acd,shape = (4,4),add_word = 'Class'):
    graph_names = list(set(graph_frame.index))
    graph_num = len(graph_names)
    fig,ax = plt.subplots(shape[1],shape[0],figsize = (12,12))
    for i,c_graph_name in enumerate(graph_names):
        c_graph = graph_frame.loc[c_graph_name,:]
        c_img = Cell_Weight_Visualization(c_graph,acd)
        sns.heatmap(c_img,center = 0,square = True, xticklabels= False, yticklabels=False,ax = ax[i//shape[0],i%shape[0]],cbar = False)
        ax[i//shape[0],i%shape[0]].axis('off')
        ax[i//shape[0],i%shape[0]].set_title(f'{add_word} {int(c_graph_name)}')
    return fig,ax