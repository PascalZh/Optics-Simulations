# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 19:13:33 2021

@author: pascalzhang
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import exp, trapz, pi
from matplotlib.widgets import Button, Slider
from matplotlib import animation
import matplotlib.lines as mlines
from numba import prange, njit, jit

N = 1000
lam = 1550e-9
k = 2 * pi / lam
L = 200 * lam
a = 25 * lam
sigma_noise = 0.005

xlim = [-a, a]
x = np.linspace(*xlim, N)


def make_iteration_equation():
    @njit(parallel=True)
    def iteration_equation(u, u_next):
        for i in prange(u_next.shape[-1]):
            u_next[i] = trapz(exp(-1j*k*((x[i] - x)**2)/(2*L)) * u, x=x)
        u_next = np.sqrt(1j/lam/L*exp(-1j*k*L)) * u_next
    return iteration_equation


iteration_equation = make_iteration_equation()


def rand(*args):
    return np.random.randn(*args)


u0_one = lambda: np.ones(shape=x.shape, dtype='complex128')
u0_3th_poly = lambda: ((x / a * 1.5) ** 3).astype('complex128')
u0_sin = lambda: (np.sin(x / a * 2 * np.pi)).astype('complex128')
u0_x = lambda: (x / a).astype('complex128')

u0 = u0_sin

us = [u0(), u0()]

fig = plt.figure('平行平面腔：自再现模', dpi=300, tight_layout=False)
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)


def plot_all(u, noise, iter_cnt, color='blue'):
    ret_artists = []
    n = x.shape[-1]
    u_angle = np.angle(u)
    u_angle = u_angle - u_angle[(n + 2) // 2]
    u_angle = (u_angle + pi) % (2*pi) - pi

    ret_artists.append(ax1.plot(x, np.abs(u), color=color)[0])
    if noise is not None:
        ret_artists.append(ax1.plot(x, np.abs(noise), linewidth=0.2,
                                    color='gray')[0])
        ret_artists.append(ax1.text(0, 0.8, f"{iter_cnt}th iteration",
                                    ha="center", va="center", size=8,
                                    bbox=bbox_props))

    ret_artists.append(ax2.plot(x, np.rad2deg(u_angle), color=color)[0])
    return ret_artists


iter_cnt = 0
artists_u0 = plot_all(us[0], None, 0, 'cyan')
artists = []


def prepare_animation():

    def animate(frame_number):
        global us, artists, iter_cnt
        iteration_equation(us[0], us[1])
        us[1] = us[1] / np.max(np.abs(us[1]))
        noise = np.zeros(*x.shape)
        # noise = sigma_noise * rand(*x.shape) + 1j*sigma_noise * rand(*x.shape)
        # us[1] = us[1] + noise
        us = [us[1], us[0]]
        iter_cnt = iter_cnt + 1

        if frame_number == 0:
            artists = plot_all(us[0], noise, iter_cnt)
            artists.append(ax1.text(0, 0.2, np.average(np.abs(np.abs(us[1]) - np.abs(us[0]))),
                                    ha="center", va="center", size=8,
                                    bbox=bbox_props))
            return artists
        else:
            return artists

    return animate


ani = animation.FuncAnimation(
    fig, prepare_animation(), frames=100, interval=1, repeat=True, blit=True)

ax1.set_xlim(xlim)
ax1.set_ylim([-0.2, 1.2])
ax2.set_xlim(xlim)
ax2.set_ylim([-180, 180])

patches = [mlines.Line2D([], [], color=c, label=l) for c, l in
           [('blue', '$|u(x)|$'), ('gray', '$|noise|$'), ('cyan', '$|u_0(x)|$')]]
ax1.legend(handles=patches, loc='upper right')

patches = [mlines.Line2D([], [], color=c, label=l)
           for c, l in [('blue', '$\\arg u(x)$'), ('cyan', '$\\arg u_0(x)$')]]
ax2.legend(handles=patches, loc='upper right')


def suptitle():
    fig.suptitle((r"$a = %.1f\lambda, L = %.1f\lambda" +
                  r", N=\frac{a^2}{L\lambda}=%.2f, " +
                  r"\left(\frac{L}{a}\right)^2=%.2f$") %
                 (a/lam, L/lam, a**2/L/lam, (L/a)**2))


suptitle()

plt.subplots_adjust(right=0.82)
ax_z = plt.axes([0.86, 0.2, 0.03, 0.6], facecolor='lightgoldenrodyellow')
ax_z2 = plt.axes([0.92, 0.2, 0.03, 0.6], facecolor='lightgoldenrodyellow')
allowed_amplitudes = np.arange(500) + 1

s_z = Slider(
    ax_z, "a (λ)", allowed_amplitudes[0], allowed_amplitudes[-1],
    valinit=int(a / lam),
    valstep=allowed_amplitudes,
    color="green",
    orientation="vertical"
)
s_z2 = Slider(
    ax_z2, "L (λ)", allowed_amplitudes[0], allowed_amplitudes[-1],
    valinit=int(L / lam),
    valstep=allowed_amplitudes,
    color="green",
    orientation="vertical"
)


def update(val):
    global a, us, xlim, x, iteration_equation, iter_cnt, artists_u0, L
    ani.pause()

    iter_cnt = 0
    a = s_z.val * lam
    L = s_z2.val * lam
    xlim = [-a, a]
    x = np.linspace(*xlim, N)
    ax1.set_xlim(xlim)
    ax2.set_xlim(xlim)
    us = [u0(), u0()]
    for art in artists_u0:
        art.remove()
    artists_u0 = plot_all(us[0], None, 0, 'cyan')

    suptitle()

    iteration_equation = make_iteration_equation()

    ani.resume()


s_z.on_changed(update)
s_z2.on_changed(update)

plt.show()
