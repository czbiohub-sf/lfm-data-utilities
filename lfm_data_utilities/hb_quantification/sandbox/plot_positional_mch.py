# -*- coding: utf-8 -*-
""" Plot estimated Hb metric against clinical Hb metric
Author: Michelle Khoo (@mwlkhoo)
Date: 2025.07

Plots data from the .csv output of get_mch.py
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


plt.hist2d(x, y, bins=30, cmap='viridis') # bins controls the number of bins, cmap sets the color map
plt.colorbar(label='Count') # Add a color bar to indicate the count/density
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.title('2D Histogram of X and Y')
plt.show()