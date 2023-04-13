'''
This script will learn basic parameters of umap function

'''
#%%


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import umap
# from Decorators import Timer

sns.set(style='white', context='poster', rc={'figure.figsize':(14,10)})

np.random.seed(42)
data = np.random.rand(800, 4)
fit = umap.UMAP()
u = fit.fit_transform(data)
#%%
plt.switch_backend('webAgg')
plt.scatter(u[:,0], u[:,1], c=data)
plt.title('UMAP embedding of random colours')
plt.show()

#%% Define plot method,
def draw_umap(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', title=''):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,# Use how much neighbor.
        min_dist=min_dist,# min dist 
        n_components=n_components,# drop to how many dim?
        metric=metric# distance method
    )
    u = fit.fit_transform(data)
    fig = plt.figure()
    if n_components == 1:
        ax = fig.add_subplot(111)
        ax.scatter(u[:,0], range(len(u)), c=data)
    if n_components == 2:
        ax = fig.add_subplot(111)
        ax.scatter(u[:,0], u[:,1], c=data)
    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(u[:,0], u[:,1], u[:,2], c=data, s=100)
    plt.title(title, fontsize=18)
    

