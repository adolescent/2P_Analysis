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
    
    plt.clf()
    plt.switch_backend('webAgg')
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
    plt.show()
    
#%% smaller n neighbor means they have 
n = 100
draw_umap(n_neighbors=n, title='n_neighbors = {}'.format(n))
#%%
# The min_dist parameter controls how tightly UMAP is allowed to pack points together.
d = 0.5
draw_umap(min_dist=d, title='min_dist = {}'.format(d))
#%% n_components provide how much dim will you down to. From 1-3 is visualizable.
draw_umap(n_components=3, title='n_components = 3')
# for cluster task, bigger dimension might be useful.


#%% Numba is a very effective coding method fo python, but can only be used on numpy.
import numba
import random
from Decorators import Timer


@Timer
# @numba.jit
def monte_carlo_pi_Raw(n_samples: int):
    acc = 0
    for i in range(n_samples):
        x = random.random()
        y = random.random()
        if (x**2 + y**2) < 1.0:
            acc += 1
    return 4.0 * acc / n_samples

@Timer
@numba.jit
def monte_carlo_pi_Numba(n_samples: int):
    acc = 0
    for i in range(n_samples):
        x = random.random()
        y = random.random()
        if (x**2 + y**2) < 1.0:
            acc += 1
    return 4.0 * acc / n_samples

N = 10000000
monte_carlo_pi_Raw(N)
monte_carlo_pi_Numba(N)

#%% UMAP support the usage of change 

@numba.jit
def red_channel_dist(a,b):
    return np.abs(a[0] - b[0])


@numba.jit
def hue(r, g, b):
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    delta = cmax - cmin
    if cmax == r:
        return ((g - b) / delta) % 6
    elif cmax == g:
        return ((b - r) / delta) + 2
    else:
        return ((r - g) / delta) + 4

# @numba.jit
@numba.njit
def hue_dist(a, b):
    diff = (hue(a[0], a[1], a[2]) - hue(b[0], b[1], b[2])) % 6
    if diff < 0:
        return diff + 6
    else:
        return diff
    
m = red_channel_dist
name = m if type(m) is str else m.__name__
draw_umap(n_components=2, metric=m, title='metric = {}'.format(name))
    