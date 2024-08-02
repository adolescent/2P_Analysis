'''
This script will plot a demostration graph of example cells.
'''
#%%
import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib

# Create a 10x10 figure at 300 dpi
fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

# Set the plot dimensions
ax.set_xlim(0,7)
ax.set_ylim(0,7)

# Draw the lines between the grid points
line_color = 'lightgray'  # Adjust the line color as needed
line_width = 1  # Adjust the line width as needed
for i in range(4):
    for j in range(4):
        for k in range(4):
            for l in range(4):
                # c_color = matplotlib.colors.hsv_to_rgb([2/3,np.random.rand()/3,1])
                c_color = matplotlib.colors.hsv_to_rgb([2/3,0,0.8])
                ax.plot([i * 10 / 6 + 1, k * 10 / 6 + 1], [j * 10 / 6 + 1, l * 10 / 6 + 1],
                        color=c_color, linewidth=2)

# Draw the 49 filled circles in gray color on top of the lines
# circle_radius = 0.36  # Adjust the radius as needed
# for i in range(4):
#     for j in range(4):
#         x = i * 10 / 6 + 1
#         y = j * 10 / 6 + 1
#         circle = plt.Circle((x, y), circle_radius, fill=True, color='gray',zorder=10)
#         ax.add_artist(circle)

# plot circles.
tri_size_u = 0.5
tri_size_lr = 0.25
for i in range(4):
    for j in range(4):
        center = [i*10/6+1,j*10/6+1]
        vertex1 = (center[0]-tri_size_lr, center[1]-tri_size_u/3)
        vertex2 = (center[0]+tri_size_lr, center[1]-tri_size_u/3)
        vertex3 = (center[0], center[1]+tri_size_u*2/3)
        ax.fill([vertex1[0], vertex2[0], vertex3[0]], 
        [vertex1[1], vertex2[1], vertex3[1]],
        color='gray', linewidth=0,zorder = 10)

# Hide the x and y axes
ax.set_axis_off()

# Display the plot
# plt.show()
# plt.savefig('D:\_GoogleDrive_Files\#Figs\Model_Demo\Point_Only.png', transparent=True)