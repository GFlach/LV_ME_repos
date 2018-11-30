# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 21:13:12 2018

@author: Admin_1
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# set up a figure 
fig = plt.figure(figsize=(10,10))

# set up the axes for the subplot
ax = fig.add_subplot(1, 1, 1, projection='3d')

# plot a 3D surface like in the example mplot3d/surface3d_demo
X = np.arange(-10, 10, 0.25)
Y = np.arange(-10, 10, 0.25)
X, Y = np.meshgrid(X, Y)
mu = -2
sig = 1.5
Z1 = 1/(sig*np.sqrt(2*np.pi))*np.exp(-(X - mu)**2/(2*sig**2))*\
        1/(sig*np.sqrt(2*np.pi))*np.exp(-(Y - mu)**2/(2*sig**2))
mu = 2
sig = 2
Z2 = 1/(sig*np.sqrt(2*np.pi))*np.exp(-(X - mu)**2/(2*sig**2))*\
        1/(sig*np.sqrt(2*np.pi))*np.exp(-(Y - mu)**2/(2*sig**2))
Z = 4*Z1 + 5*Z2
ax.plot_surface(X, Y, Z)
ax.contourf(X, Y, Z, zdir='z', offset=-2,
        levels=np.linspace(0,0.3,1000),cmap=plt.cm.jet)
ax.contourf(X, Y, Z, zdir='x', offset=-10, cmap=plt.cm.jet)
ax.contourf(X, Y, Z, zdir='y', offset=10, cmap=plt.cm.jet)
ax.set_zlim(-2, 1)

plt.show()