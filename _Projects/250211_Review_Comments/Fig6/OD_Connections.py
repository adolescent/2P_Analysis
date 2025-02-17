'''
Plot LR band, annotation and colorize later, just plot lines.

We use clever method through plt.plot, no need for cv2.
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

# plot inner line
ax.plot([3,3], [3,9],color='black', linewidth=inner_width)
ax.plot([5,5], [3,9],color='black', linewidth=outer_width)
ax.plot([7,7], [3,9],color='black', linewidth=inner_width)

## and plot connections here.
# connect LE and LE.
hue_order = ['r','b']
for i in range(2):
    c_deep = 0.5*(i+1)
    start = [2+i*2,3]
    end = [6+i*2,3]
    # plot 3 lines.
    ax.plot([start[0],start[0]],[start[1],start[1]-c_deep],color = hue_order[i],linewidth = connect_width)
    ax.plot([end[0],end[0]],[end[1],end[1]-c_deep],color = hue_order[i],linewidth = connect_width)
    ax.plot([start[0],end[0]],[start[1]-c_deep,start[1]-c_deep],color = hue_order[i],linewidth = connect_width)

# Hide the x and y axes and display graph
ax.set_axis_off()
plt.show()