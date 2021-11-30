# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 11:39:43 2021

@author: pascalzhang
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

C = 0
T0 = 1.2
m = 3
beta2 = 0.2
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

def U(z):
    U0 = np.exp(- (1 + C * 1j)/2 * ((T/T0)**(2*m)))
    
    U0_ = np.fft.fft(U0)
    dispersion = np.exp(1j / 2 *beta2*(w**2)*z+ 1j / 6 * beta3 * (w**3)*z)

    return np.fft.ifft(U0_ * dispersion), U0_ * dispersion

fig = plt.figure(dpi=300)
ax1 = fig.add_subplot(211)
plt.subplots_adjust(bottom=0.25)


z = 0.
sp, _ = U(z)


l1, = ax1.plot(T, (sp * sp.conj()).real, label='$|U(0)|^2$')
plt.legend()


# Calculate U at z
z = 1 * min(LD, LD_)
sp, Uz_ = U(z)


l2, = ax1.plot(T, (sp * sp.conj()).real, label='$|U(z)|^2$')
ax1.legend()
ax1.set_title(f"Time domain, $L_D$={LD}, $L'_D$ = {LD_}" )
ax1.set_ylim((-0.5, 1.5))

ax2 = fig.add_subplot(212)
l3, = ax2.plot(freq, np.abs(Uz_))
ax2.tick_params(axis='y', labelcolor=l3.get_c())
ax2twinx = ax2.twinx()

l4, = ax2twinx.plot(freq, np.angle(Uz_), linewidth=0.5, color='green')
ax2twinx.tick_params(axis='y', labelcolor=l4.get_c())
ax2.legend([r'$\|\tilde{U}(z)\|$', r'$\angle\tilde{U}(z)$'])

slider_bkd_color = 'lightgoldenrodyellow'
ax_z = plt.axes([0.10, 0.1, 0.65, 0.03], facecolor=slider_bkd_color)
ax_C = plt.axes([0.10, 0.15, 0.65, 0.03], facecolor=slider_bkd_color)

# create the sliders
s_z = Slider(
    ax_z, "z", 0., 4 * min(LD, LD_),
    valinit=z,
    color="green"
)
s_C = Slider(
    ax_C, "C", -10, 10,
    valinit=C,
    color="green"
)

def update(val):
    global C, z
    z = s_z.val
    C = s_C.val
    sp, Uz_ = U(z)
    l2.set_ydata((sp * sp.conj()).real)
    l3.set_ydata(np.abs(Uz_))
    l4.set_ydata(np.angle(Uz_))
    fig.canvas.draw_idle()
s_z.on_changed(update)
s_C.on_changed(update)

ax_reset = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(ax_reset, 'Reset', color=slider_bkd_color, hovercolor='0.975')

def reset(event):
    s_z.reset()
    s_C.reset()
button.on_clicked(reset)

plt.show()