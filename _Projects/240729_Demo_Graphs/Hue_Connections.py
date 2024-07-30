'''
This will Plot blob on graph, used for hue hyper column
'''

#%%
import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib


# Create a 10x10 figure at 300 dpi
fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

outer_width = 3
inner_width = 2
connect_width = 1

# Set the plot dimensions
ax.set_xlim(0,10)
ax.set_ylim(0,10)


# plot outline
ax.plot([1,9], [3,3],color='black', linewidth=outer_width) # [x1,x2],[y1,y2] seq.
ax.plot([1,9], [9,9],color='black', linewidth=outer_width)
ax.plot([1,1], [3,9],color='black', linewidth=outer_width)
ax.plot([9,9], [3,9],color='black', linewidth=outer_width)


# plot 3 blobs
blob_width = 0.8
blob_depth = 2.5
inter = 2.5
for i in range(3):
    start = 2.2+i*inter
    ax.plot([start,start], [9-blob_depth,9],color='black', linewidth=inner_width)
    ax.plot([start+blob_width,start+blob_width], [9-blob_depth,9],color='black', linewidth=inner_width)
    ax.plot([start,start+blob_width], [9-blob_depth,9-blob_depth],color='black', linewidth=inner_width)

# plot connections.
depth = 0.3
for i in range(3):
    start = 2.6+inter*i
    ax.plot([start,start], [9,9+depth],color='r', linewidth=connect_width)
ax.plot([2.6,2.6+inter*2], [9+depth,9+depth],color='r', linewidth=connect_width)


# Hide the x and y axes and display graph
ax.set_axis_off()
plt.show()