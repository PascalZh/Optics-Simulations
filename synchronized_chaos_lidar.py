# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 10:03:12 2021

Reference article: [1]W.-T. Wu, Y.-H. Liao, and F.-Y. Lin, “Noise suppressions in synchronized chaos lidars,” Opt. Express, vol. 18, no. 25, p. 26155, Dec. 2010, doi: 10.1364/OE.18.026155.

@author: pascalzhang
"""

import numpy as np
from numpy import sin, cos, pi
from numpy.random import default_rng
from ddeint import ddeint
import time
import numba

import matplotlib.pyplot as plt


def gen_model(eta_OF_T=0, eta_OF_R=0, eta_OF_C=0, eta_OEF_T=0, eta_OEF_R=0, eta_OEF_C=0, tau_T=0, tau_R=0, tau_C=0, sigma_1=0, sigma_2=0, m=0):
    c = 299792458.0  # the speed of light in vaccum
    omega_T = 2 * pi * c / 1.55e-6
    omega_R = omega_T

    # laser parameters
    b = 4  # line width enhancement factor
    gamma_n = 0.667e9  # differential carrier relaxation rate
    gamma_p = 1.2e9  # nonlinear carrier relaxation rate
    gamma_s = 1.458e9  # spontaneous carrier relaxation rate
    gamma_c = 2.4e11  # cavity decay rate
    J = 0.333  # normalized dimensionless injection current parameter
    delta_omega = 0  # angular frequency detuning between the Tx and Rx

    rng = default_rng()

    def model(Y, t):

        a_T, phi_T, n_T, a_R, phi_R, n_R = Y(t)

        tmp1_T = (gamma_c * gamma_n / gamma_s / J *
                  n_T - gamma_p * (2 * a_T + a_T ** 2))
        tmp1_R = (gamma_c * gamma_n / gamma_s / J *
                  n_R - gamma_p * (2 * a_R + a_R ** 2))

        a_C_delayed = Y(t - tau_C)[0] + sigma_1 * rng.standard_normal()
        phi_C_delayed = Y(t - tau_C)[1] + m * sigma_2 * rng.standard_normal()

        channel_a = eta_OF_C * \
            (1 + a_C_delayed) * cos(phi_C_delayed -
                                    phi_R + omega_T * tau_C - delta_omega * t)

        channel_phi = eta_OF_C * (1 + a_C_delayed) / (1 + a_R) * \
            sin(phi_C_delayed - phi_R + omega_T * tau_C - delta_omega * t)

        channel_n = eta_OEF_C * gamma_s * \
            (J + 1) * (1 + 2*a_C_delayed) + a_C_delayed**2

        a_T_dot = 0.5 * tmp1_T * (1 + a_T) + eta_OF_T * (1 + Y(t - tau_T)[0]) * cos(Y(t - tau_T)[1] - phi_T + omega_T * tau_T)

        phi_T_dot = -b/2 * tmp1_T + \
            eta_OF_T * (1 + Y(t - tau_T)[0]) / (1 + a_T) * \
            sin(Y(t - tau_T)[1] - phi_T + omega_T * tau_T)

        n_T_dot = -gamma_s * n_T - gamma_n * (1 + a_T)**2 * n_T - gamma_s * J * (2*a_T + a_T**2) + gamma_s * gamma_p / gamma_c * J * (2*a_T + a_T**2) * (1 + a_T) ** 2 + \
            eta_OEF_T * gamma_s * (J + 1) * \
            (1 + 2*Y(t - tau_T)[0]) + Y(t - tau_T)[0]**2

        a_R_dot = 0.5 * tmp1_R * (1 + a_R) + eta_OF_R * (1 + Y(t - tau_R)[3]) * cos(Y(t - tau_R)[4] - phi_R + omega_R * tau_R) + \
            channel_a

        phi_R_dot = - b / 2 * tmp1_R + \
            eta_OF_R * (1 + Y(t - tau_R)[3]) / (1 + a_R) * sin(Y(t - tau_R)[4] - phi_R + omega_R * tau_R) + \
            channel_phi

        n_R_dot = -gamma_s * n_R - gamma_n * (1 + a_R)**2 * n_R - gamma_s * J * (2*a_R + a_R**2) + gamma_s * gamma_p / gamma_c * J * (2*a_R + a_R**2) * (1 + a_R) ** 2 + \
            eta_OEF_R * gamma_s * (J + 1) * (1 + 2*Y(t - tau_R)[3]) + Y(t - tau_R)[3]**2 + \
            channel_n

        return np.array([a_T_dot, phi_T_dot, n_T_dot, a_R_dot, phi_R_dot, n_R_dot])

    return model


model1 = gen_model(eta_OF_T=0.2, eta_OF_R=0.2, eta_OF_C=0.2,
                   tau_T=9.5e-9, tau_R=9.5e-9, tau_C=15.5e-9,
                   sigma_1=1)

h = 1e-12
t_max = 10e-9
tt = np.linspace(0, t_max, int(t_max / h / 2) * 2) 
N = tt.shape[-1]

start = time.time()
yy = ddeint(model1, lambda t: np.array([0, 0, 0, 0, 0, 0]), tt)
print(time.time() - start)

a_T = yy[:, 0]
a_R = yy[:, 3]

fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

ax1.plot(tt, np.abs(a_T))
ax2.plot(tt, np.abs(a_R))
# ax3.plot(tt, np.correlate(a_T, a_T, mode='same'))
# ax4.plot(tt, np.correlate(a_R, a_R, mode='same'))
ax3.plot(np.fft.fftfreq(tt.shape[-1])[:N//2], 20 * np.log10(np.abs(np.fft.fft(a_T)))[:N//2])
ax3.set_xlabel('frequency')
ax4.plot(np.fft.fftfreq(tt.shape[-1])[:N//2], 20 * np.log10(np.abs(np.fft.fft(a_R)))[:N//2])
ax4.set_xlabel('frequency')

plt.show()
