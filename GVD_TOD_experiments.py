# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 11:39:43 2021

@author: pascalzhang
"""

import numpy as np
import matplotlib.pyplot as plt
from putils.plotutils import PlotUI_Sliders, SliderParam

C = 0
T0 = 1.2
m = 3
beta2 = 0.
beta3 = 0.

LD = T0**2 / np.abs(beta2) if beta2 != 0. else float('inf')  # GVD length
LD_ = T0**3 / np.abs(beta3) if beta3 != 0. else float('inf')  # TOD length
z = 0

l1 = None  # line that z = 0
l2 = None  # line that z > 0
N = 1000
half_window_width = 8 * T0
T = np.linspace(-half_window_width, half_window_width, N)
fs = 1/(T[-1] - T[0])*N
freq = np.fft.fftfreq(T.shape[-1], d=1/fs)
w = 2 * np.pi * freq


def z_limits(): return min(LD, LD_, 200)


def U(z):
    U0 = np.exp(- (1 + C * 1j)/2 * ((T/T0)**(2*m)))

    U0_ = np.fft.fft(U0)
    dispersion = np.exp(1j/2 * beta2 * (w**2)*z + 1j/6 * beta3 * (w**3)*z)

    return np.fft.ifft(U0_ * dispersion), U0_ * dispersion


ui = PlotUI_Sliders(
    2,
    (SliderParam("z", 0, 4 * z_limits(), z_limits()), SliderParam("C", -10, 10, C)),
    (SliderParam("$\\beta_2$", -1, 1, beta2), SliderParam('$\\beta_3$', -1, 1, beta3))
)
ax1, ax2 = ui.axes[0:2]


sp, _ = U(0)
l1, = ax1.plot(T, (sp * sp.conj()).real, label='$|U(0)|^2$')


# Calculate U at z
sp, Uz_ = U(z_limits())
l2, = ax1.plot(T, (sp * sp.conj()).real, label='$|U(z)|^2$')

ax1.legend()
ax1.set_title(f"Time domain, $L_D$={LD}, $L'_D$ = {LD_}")
ax1.set_ylim((-0.5, 1.5))

l3, = ax2.plot(freq, np.abs(Uz_))
ax2.tick_params(axis='y', labelcolor=l3.get_c())
ax2twinx = ax2.twinx()

l4, = ax2twinx.plot(freq, np.angle(Uz_), linewidth=0.5, color='green')
ax2twinx.tick_params(axis='y', labelcolor=l4.get_c())
ax2.legend([r'$\|\tilde{U}(z)\|$'], loc='upper left')
ax2twinx.legend([r'$\angle\tilde{U}(z)$'], loc='upper right')


def update(val):
    global C, z, beta2, beta3
    z = ui.sliders[0].val
    C = ui.sliders[1].val
    beta2 = ui.sliders[2].val
    beta3 = ui.sliders[3].val
    sp, Uz_ = U(z)
    l2.set_ydata((sp * sp.conj()).real)
    l3.set_ydata(np.abs(Uz_))
    l4.set_ydata(np.angle(Uz_))


ui.on_changed(update)

plt.show()
