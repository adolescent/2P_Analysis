
#%%

import cv2
import imageio
import numpy as np

# Read the graph images using cv2 (replace with your own image paths)
graph1_img = cv2.imread(r'C:\Users\admin\Desktop\gifs\1.jpg')
graph2_img = cv2.imread(r'C:\Users\admin\Desktop\gifs\2.jpg')
graph3_img = cv2.imread(r'C:\Users\admin\Desktop\gifs\3.jpg')

# Create an empty list to store the frames
frames = []

# Add the graph frames to the list
frames.append(graph1_img)
frames.append(np.zeros_like(graph1_img))  # Add a blank frame

frames.append(graph2_img)
frames.append(np.zeros_like(graph2_img))  # Add a blank frame

frames.append(graph3_img)
frames.append(np.zeros_like(graph3_img))  # Add a blank frame

# Save the frames as a GIF
imageio.mimsave('graph_animation.gif', frames, format='GIF',loop = 0,duration = [300,80,300,80,300,80])
# imageio.mimsave('graph_animation.gif', frames, format='GIF',loop = 0,duration = 380)