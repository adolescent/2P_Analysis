# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 16:25:07 2020

@author: ZR
Test Run File
Do not save.
"""
#%%

# Code source: Gaël Varoquaux
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from sklearn import decomposition
from sklearn import datasets

np.random.seed(5)

centers = [[1, 1], [-1, -1], [1, -1]]
iris = datasets.load_iris()
X = iris.data
y = iris.target

fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=40, azim=110)

plt.cla()
pca = decomposition.PCA()
pca.fit(X)
X = pca.transform(X)

for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
    ax.text3D(X[y == label, 0].mean(),
              X[y == label, 1].mean() + 1.5,
              X[y == label, 2].mean(), name,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [1, 2, 0]).astype(np.float)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.nipy_spectral,
           edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

plt.show()

#%%

import matplotlib.pyplot as plt
import numpy as np
a = np.random.multivariate_normal([0,0],cov = [[10.5246,9.6313],[9.6313,11.3203]],size = 5000)
#transfer_matrix = np.array([[0.707,-0.707],[0.707,0.707]])
#test = np.dot(a,transfer_matrix)
plt.ylim([-15,15])
plt.xlim([-20,20])
plt.xlabel('X')
plt.ylabel('Y')
plt.scatter(a[:,0],a[:,1],s = 1)# s 是点大小
plt.arrow(0,0,0.6906*10,0.7231*10,head_width=1, head_length=1, shape="full",fc='red',ec='red')
plt.arrow(0,0,-0.7215*2,0.6923*2,head_width=1, head_length=1, shape="full",fc='red',ec='red')
#plt.scatter(test[:,0],test[:,1])
#%%
from sklearn import decomposition
pca = decomposition.PCA()
pca.fit(a)
b = pca.transform(a)
plt.ylim([-15,15])
plt.xlim([-20,20])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.scatter(b[:,0],b[:,1],s = 1)
#%%
