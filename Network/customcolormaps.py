# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 20:30:37 2021

@author: BlankenN
"""

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def rgb_list_to_cmap(colorlist_RGB,colornodes):
    
    # Convert color list to numpy:
    colorlist_RGB = np.array(colorlist_RGB)

    # Convert RGB values to values  between 0 and 1:
    colorlist     = colorlist_RGB/255
    
    cdict = {'red':   [[colornodes[k],  colorlist[k][0], colorlist[k][0]] 
                       for k in range(len(colornodes))],
             'green': [[colornodes[k],  colorlist[k][1], colorlist[k][1]] 
                       for k in range(len(colornodes))],
             'blue':  [[colornodes[k],  colorlist[k][2], colorlist[k][2]] 
                       for k in range(len(colornodes))]}
    
    # Create colormap:
    cmp = LinearSegmentedColormap('testCmap', segmentdata=cdict, N=256)
    
    return cmp


def get_custom_colormap(cmap_name):
    # List of colors in RGB:
    red_colors = [[73,  0,   0  ],
                  [255, 82,  82 ],
                  [255, 220, 220]]
   
    
    blue_colors = [[0,   0,   121],
                   [0,   114, 189],
                   [188, 225, 255]]
    
    # Colormap nodes
    red_colornodes = [0, 0.6, 1.0]
    blue_colornodes = [0, 0.39, 1.0]
    
    if cmap_name == 'red':
        cmap = rgb_list_to_cmap(red_colors,  red_colornodes)
    if cmap_name == 'blue':
        cmap = rgb_list_to_cmap(blue_colors, blue_colornodes)
        
    return cmap
            

