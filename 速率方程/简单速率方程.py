# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 20:52:50 2021

G=gain coefficient, f=decay rate due to stimulated emission, b=decay rate due to
transmission/scattering, and p=pump strength. 

Reference: ./参考资料/laser_report.pdf

@author: pascalzhang
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from numba import njit
from itertools import product

t_span = [0, 200]


@njit
def f(t, y):
    q, n = y
    G = 0.5
    b = 0.2
    f = 0.5
    p = 0.2
    return np.array([G*n*q - b*q, -G*n*q-f*n+p])


fig = plt.figure(dpi=300, tight_layout=True)
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

result = integrate.solve_ivp(f, t_span, [0.3, 0.4], dense_output=True)
t = np.linspace(*t_span, 2000)
y = result.sol(t)

ax1.plot(t, y.T)
ax2.plot(y[0], y[1])

for q0 in np.linspace(0, 0.8*2.5, int(15*2.5))[1:]:
    for n0 in np.linspace(0, 0.8, 15)[1:]:
        result = integrate.solve_ivp(f, t_span, [q0, n0], dense_output=True)
        y = result.sol(t)
        # ax1.plot(t, y.T)
        ax2.plot(y[0], y[1])

ax1.legend(['q', 'n'])
ax1.set_xlabel('t')
ax2.set_title('Different initial (q, n) values')
ax2.set_xlabel('q')
ax2.set_ylabel('n')

plt.show()
