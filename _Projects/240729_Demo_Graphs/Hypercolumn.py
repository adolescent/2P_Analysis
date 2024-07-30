'''
This will just draw a hyper column brief graph.
'''

#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import cv2


#%% Plot graph
graph = np.ones(shape = (2000,2000),dtype = 'u1')*255
angle = 45*np.pi/180
# define point of start.
base_len = 1000 # horizontal size
base_height = 600# vertical size
base_width = 600# angled size
color = 0
blob_color = 172
width = 4
surface_width = 2

A = (100,600)
B = (A[0]+base_len,600)
C = (int(B[0]+base_width*np.cos(angle)),int(B[1]-base_width*np.sin(angle)))
D = (int(A[0]+base_width*np.cos(angle)),int(A[1]-base_width*np.sin(angle)))

# Plot above
graph = cv2.line(graph, A, B,color,width)
graph = cv2.line(graph, B, C,color,width)
graph = cv2.line(graph, D, C,color,width)
graph = cv2.line(graph, A, D,color,width)

# plot vertical
A1 = (A[0],A[1]+base_height)
B1 = (B[0],B[1]+base_height)
C1 = (C[0],C[1]+base_height)
graph = cv2.line(graph, A, A1,color,width)
graph = cv2.line(graph, B, B1,color,width)
graph = cv2.line(graph, C, C1,color,width)
graph = cv2.line(graph, A1, B1,color,width)
graph = cv2.line(graph, B1, C1,color,width)

# plot orien bars here.
dist = int(base_len/16)
for i in range(15):
    start = (A[0]+dist*(i+1),A[1])
    end = (A1[0]+dist*(i+1),A1[1])
    graph = cv2.line(graph, start, end,color,width)
    # add a surface divider
    if i ==7:
        graph = cv2.line(graph, start,(int(start[0]+base_width*np.cos(angle)),int(start[1]-base_width*np.sin(angle))),color,surface_width)
    
# plot OD bars here
od_x_dist = int(abs(C[0]-B[0])/4)
od_y_dist = int(abs(C[1]-B[1])/4)
for i in range(3):
    start = (B[0]+od_x_dist*(i+1),B[1]-od_y_dist*(i+1))
    end = (start[0],start[1]+base_height)
    graph = cv2.line(graph, start, end,color,width)
    # add a surface divider.
    if i == 1:
        graph = cv2.line(graph, start,(start[0]-base_len,start[1]),color,surface_width)
    
# plot color blobs on the surface.
oval_dist_angled = int(base_width/4)
oval_dist_h = base_len/7
oval_dist_h = int(oval_dist_h+oval_dist_angled*np.cos(angle))
oval_dist_v = int(oval_dist_angled*np.sin(angle))
init_oval = (int(A[0]+oval_dist_h*0.7),int(A[1]-oval_dist_v*0.6))
for i in range(4):
    for j in range(4):
        c_oval_cen = (init_oval[0]+i*oval_dist_h+j*oval_dist_v,init_oval[1]-j*oval_dist_v)
        graph = cv2.ellipse(graph,c_oval_cen,(50,25),0, 0,360, blob_color,-1)

# add orien bars on graph.
init_cen = (int((A[0]+dist*0.5)),(A[1]+100))
bar_len = 20
for i in range(16):
    c_cen = (int(init_cen[0]+dist*i),int(init_cen[1]))
    c_orien = i*22.5*np.pi/180
    bar_len_h = int(bar_len*np.cos(c_orien))
    bar_len_v = int(bar_len*np.sin(c_orien))
    graph = cv2.line(graph,c_cen,(c_cen[0]+bar_len_h,c_cen[1]+bar_len_v),color,2)
    graph = cv2.line(graph,c_cen,(c_cen[0]-bar_len_h,c_cen[1]-bar_len_v),color,2)

# show time
cv2.imshow('Hypercolumn',graph)
cv2.waitKey(5000)
cv2.destroyAllWindows()
# plt.clf()
# plt.cla()
# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5),dpi = 300)
# sns.heatmap(graph,cmap='gist_gray',center = 0,xticklabels=False,yticklabels=False,square=True,cbar= None,ax = ax)
cv2.imwrite(r'D:\_GoogleDrive_Files\#Figs\#Second_Final_Version-240710\Add1_OI_Compare\Hypercolumn.png',graph)