# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 16:25:07 2020

@author: ZR
This File include examples of PCA&PCA ppt.
Spyder is recommended to repeat graphs.
"""



#%% This part is Iris Example.

# Code source: Gaël Varoquaux
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target

fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=30, azim=100)

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
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.show()

#%% This Part is 2D random dots example.
import numpy as np
a = np.random.multivariate_normal([5,10],cov = [[10.5246,9.6313],[9.6313,11.3203]],size = 5000)

import matplotlib.pyplot as plt
plt.ylim([-5,25])
plt.xlim([-15,25])
plt.xlabel('X')
plt.ylabel('Y')
plt.scatter(a[:,0],a[:,1],s = 1)# s 是点大小
#plt.arrow(0,0,0.6906*10,0.7231*10,head_width=1, head_length=1, shape="full",fc='red',ec='red')
#plt.arrow(0,0,-0.7215*2,0.6923*2,head_width=1, head_length=1, shape="full",fc='red',ec='red')

from sklearn import decomposition
pca = decomposition.PCA()
pca.fit(a)
pc_1 = pca.components_[0,:]
pc_2 = pca.components_[1,:]
fig2 = plt.figure()
b = pca.transform(a)
plt.ylim([-15,15])
plt.xlim([-20,20])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.scatter(b[:,0],b[:,1],s = 1)
#%% This part is PCA in mixed signal example.
import numpy as np
import matplotlib.pyplot as plt
a = np.random.multivariate_normal([0,0],cov = [[10.5246,9.6313],[9.6313,11.3203]],size = 5000)
b = np.random.multivariate_normal([0,0],cov = [[20.13,0],[0,1.3]],size = 5000)
c = np.vstack((a,b))# connect 2 distribution.
plt.scatter(c[:,0],c[:,1],s = 1)
# Then plot 2 PC as arrow
# =============================================================================
# pc_1 = pca.components_[0,:]
# pc_2 = pca.components_[1,:]
# plt.arrow(0,0,pc_1[0]*8,pc_1[1]*8,head_width=1, head_length=1, shape="full",fc='red',ec='red')
# plt.arrow(0,0,pc_2[0]*3,pc_2[1]*3,head_width=1, head_length=1, shape="full",fc='red',ec='red')
# 
# =============================================================================
from sklearn import decomposition
fig2 = plt.figure()
pca = decomposition.PCA()
pca.fit(c)
c_trans = pca.transform(c)
plt.ylim([-15,15])
plt.xlim([-20,20])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.scatter(c_trans[:,0],c_trans[:,1],s = 1)
#%% This part is ICA perform on mixed signal example.
import numpy as np
import matplotlib.pyplot as plt
# #############################################################################
# Generate sample data
rng = np.random.RandomState(42)
S = rng.standard_t(1.5, size=(20000, 2))
S[:, 0] *= 2.
# Mix data
A = np.array([[1, 1], [0, 2]])  # Mixing matrix
X = np.dot(S, A.T)  # Generate observations
plt.xlim(-30, 30)
plt.ylim(-30, 30)
plt.scatter(X[:,0],X[:,1],s = 1)
# Till now, X is the distribution.

from sklearn import decomposition
ica = decomposition.FastICA()
ica.fit(X)
ic_1 = ica.mixing_[:,0]
ic_2 = ica.mixing_[:,1]
plt.arrow(0,0,ic_1[0]*0.005,ic_1[1]*0.005,head_width=1, head_length=1, shape="full",fc='red',ec='red')
plt.arrow(0,0,ic_2[0]*0.003,ic_2[1]*0.003,head_width=1, head_length=1, shape="full",fc='red',ec='red')
trans_c = ica.transform(X)
fig2 = plt.figure()
plt.scatter(trans_c[:,0],trans_c[:,1],s = 1)
plt.xlim(-0.2,0.2)
plt.ylim(-0.2,0.2)
#%% This is a decorator test.
def use_log(func): 
    def wrapper(*args, **kwargs): 
        print("%s is running" % func.__name__)
        return func(*args)
    return wrapper
@use_log
def foo(): 
    print("i am foo")
foo()
# A timer we want to use.
import time    
def Timer(func):
    def core_func(*args,**kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        print("%s is running" % func.__name__)
        end_time = time.time()
        print('Time Cost: %ss' % (end_time-start_time))
        return result
    return core_func
@Timer
def Sleep():
    time.sleep(3)
Sleep()
#%% Test Vector Autoregression model.
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
mdata = smapi.datasets.macrodata.load_pandas().data
dates = mdata[['year', 'quarter']].astype(int).astype(str)
quarterly = dates["year"] + "Q" + dates["quarter"]
from statsmodels.tsa.base.datetools import dates_from_str
quarterly = dates_from_str(quarterly)
mdata = mdata[['realgdp','realcons','realinv']]
mdata.index = pd.DatetimeIndex(quarterly)
data = np.log(mdata).diff().dropna()
model = VAR(data)
#results = model.fit(2)# 2 order data, meaning 2 lag at most.
results = model.fit(maxlags=15, ic='aic')
results.summary()



