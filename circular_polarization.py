# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 19:45:23 2021

@author: pascalzhang
"""
import numpy as np
import matplotlib.pyplot as plt


x_base = np.array([0, 1])
y_base = np.array([1, 0])

z = np.linspace(0, 4, 1000)
k = 2 * np.pi * 1

E = np.vstack((np.sin(k * z), np.zeros(*z.shape)))
E += np.vstack((np.zeros(*z.shape),  1.5 * np.sin(k * z + np.pi / 2)))

fig = plt.figure(dpi=300)
ax = fig.add_subplot(211, projection='3d')
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])

ax2 = fig.add_subplot(212)
ax.plot(E[0], E[1], z)
ax2.plot(np.sin(k * z))
plt.show()
