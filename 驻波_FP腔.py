# -*- coding: utf-8 -*-
"""
Created on Sun Feb  20 23:44:00 2022

@author: pascalzhang
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos
from matplotlib import animation

N = 10000
k = 0.5
w = 0.6
L = 5 * 0.5 / k * 2 * pi

x = np.linspace(0, L, N)

t = 0.0
def prepare_animation():

    def animate(frame_number):
        global t
        t = t + 0.1
        l = plt.plot(x, sin(k*x)*cos(w*t), color='blue')  # cos(w*t) or sin(w*t) or any trigonometric functions 
        return l

    return animate

fig = plt.figure()

ani = animation.FuncAnimation(
    fig, prepare_animation(), frames=300, interval=1000//60, repeat=True, blit=True)

plt.show()
