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
for i in range(15):
    if i == 7:
        c_width = outer_width
    else:
        c_width = inner_width
    ax.plot([1.5+i*0.5,1.5+i*0.5], [3,9],color='black', linewidth=c_width)
    


## and plot connections here.
for i in range(8):
    c_hue = matplotlib.colors.hsv_to_rgb([i*0.125,1,1])
    c_deep = 0.2*(i+1)
    start = [1.25+i*0.5,3]
    end = [5.25+i*0.5,3]
    # plot 3 lines.
    ax.plot([start[0],start[0]],[start[1],start[1]-c_deep],color = c_hue,linewidth = connect_width)
    ax.plot([end[0],end[0]],[end[1],end[1]-c_deep],color = c_hue,linewidth = connect_width)
    ax.plot([start[0],end[0]],[start[1]-c_deep,start[1]-c_deep],color = c_hue,linewidth = connect_width)

# add orientation bars on graph.
bar_len = 0.17
for i in range(16):
    c_cen = [1.25+i*0.5,6]
    print(c_cen)
    c_orien = i*22.5*np.pi/180
    bar_len_h = bar_len*np.cos(c_orien)
    bar_len_v = bar_len*np.sin(c_orien)
    ax.plot([c_cen[0],c_cen[0]+bar_len_h],[c_cen[1],c_cen[1]+bar_len_v],color = 'black',linewidth = inner_width,zorder=10)
    ax.plot([c_cen[0],c_cen[0]-bar_len_h],[c_cen[1],c_cen[1]-bar_len_v],color = 'black',linewidth = inner_width,zorder=10)


# Hide the x and y axes and display graph
ax.set_axis_off()
plt.show()

