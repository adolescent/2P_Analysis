'''
Provide multiple plot tools to save code lines.
'''

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm


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