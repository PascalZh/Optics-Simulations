# Reference article: [1]W.-T. Wu, Y.-H. Liao, and F.-Y. Lin, “Noise suppressions in synchronized chaos lidars,” Opt. Express, vol. 18, no. 25, p. 26155, Dec. 2010, doi: 10.1364/OE.18.026155.

using DifferentialEquations
using Plots
using FFTW

const c = 299792458.0
const ω_T = 2 * pi * c / 1.55e-6
const ω_R = ω_T

# laser parameters
const b = 4  # line width enhancement factor
const γ_n = 0.667e9  # differential carrier relaxation rate
const γ_p = 1.2e9  # nonlinear carrier relaxation rate
const γ_s = 1.458e9  # spontaneous carrier relaxation rate
const γ_c = 2.4e11  # cavity decay rate
const J = 0.333  # normalized dimensionless injection current parameter
const Δω = 0  # angular frequency detuning between the Tx and Rx

function s_clidar_model(du, u, h, p, t)
  η_OF_T, η_OF_R, η_OF_C, η_OEF_T, η_OEF_R, η_OEF_C, τ_T, τ_R, τ_C, σ_1, m = p

  a_T, ϕ_T, n_T, a_R, ϕ_R, n_R = u

  tmp1_T = γ_c * γ_n / γ_s / J * n_T - γ_p * (2a_T + a_T^2)
  # tmp1_R = γ_c * γ_n / γ_s / J * n_R - γ_p * (2a_R + a_R^2)

  # a_C_delayed = h(p, t - τ_C)[1] + σ_1 * randn()
  # ϕ_C_delayed = h(p, t - τ_C)[2] + m * (2π * rand() - π)

  # channel_a = η_OF_C * (1 + a_C_delayed) * cos(ϕ_C_delayed - ϕ_R + ω_T * τ_C)

  # channel_ϕ = η_OF_C * (1 + a_C_delayed) / (1 + a_R) * sin(ϕ_C_delayed - ϕ_R + ω_T * τ_C)

  # channel_n = η_OEF_C * γ_s * (J + 1) * (1 + 2a_C_delayed) + a_C_delayed^2

  du[1] = 0.5tmp1_T*(1 + a_T) + η_OF_T*(1 + h(p, t - τ_T)[1]) * cos(h(p, t - τ_T)[2] - ϕ_T + ω_T*τ_T)

  du[2] = -0.5b*tmp1_T + η_OF_T*(1 + h(p, t - τ_T)[1])/(1 + a_T)*sin(h(p, t - τ_T)[2] - ϕ_T + ω_T*τ_T)

  du[3] = -γ_s*n_T - γ_n*(1 + a_T)^2*n_T - γ_s*J*(2a_T + a_T^2) + γ_s*γ_p/γ_c*J*(2a_T + a_T^2)*(1 + a_T)^2 + η_OEF_T * γ_s * (J + 1) * (1 + 2h(p, t - τ_T)[1]) + h(p, t - τ_T)[1]^2

  # du[4] = 0.5tmp1_R * (1 + a_R) + η_OF_R * (1 + h(p, t - τ_R)[4]) * cos(h(p, t - τ_R)[5] - ϕ_R + ω_R * τ_R) + channel_a

  # du[5] = -0.5b * tmp1_R + η_OF_R * (1 + h(p, t - τ_R)[4]) / (1 + a_R) * sin(h(p, t - τ_R)[5] - ϕ_R + ω_R * τ_R) + channel_ϕ

  # du[6] = -γ_s * n_R - γ_n * (1 + a_R)^2 * n_R - γ_s * J * (2a_R + a_R^2) + γ_s * γ_p / γ_c * J * (2a_R + a_R^2) * (1 + a_R)^2 + η_OEF_R * γ_s * (J + 1) * (1 + 2h(p, t - τ_R)[4]) + h(p, t - τ_R)[4]^2 + channel_n

end

τ_T = 9.5e-9
τ_R = 9.5e-9
τ_C = 15.5e-9
lags = [τ_T, τ_R, τ_C]

η_OF_T, η_OF_R, η_OF_C = 0.2, 0.2, 1.3
η_OEF_T, η_OEF_R, η_OEF_C = 0, 0, 0
σ_1, m = 0, 0

h(p, t) = zeros(6)
u0 = [0, 0, 0, 0, 0, 0]
tspan = (0, 200e-9)
p = (η_OF_T, η_OF_R, η_OF_C, η_OEF_T, η_OEF_R, η_OEF_C, τ_T, τ_R, τ_C, σ_1, m)

prob = DDEProblem(s_clidar_model, u0, h, tspan, p; constant_lags = lags)
alg = MethodOfSteps(Tsit5())
@time sol = solve(prob, alg; dt=0.1e-12, adaptive=true)

fs = (sol.t[end] - sol.t[1]) / (length(sol.t) - 1)
freqs = fftfreq(length(sol.t), fs)
freqs = fftshift(freqs)

a_T = Float64.([u[1] for u in sol.u])
a_R = Float64.([u[4] for u in sol.u])
A_T = fftshift(fft(a_T))
A_R = fftshift(fft(a_R))

p1 = plot(sol.t, a_T, label="\$a_T\$")
p2 = plot(sol.t, a_R, label="\$a_R\$")
p3 = plot(freqs, abs.(A_T), label="\$\\hat{a}_T\$")
p4 = plot(sol)
# p4 = plot(freqs, abs.(A_R), label="\$\\hat{a}_R\$")

plt = plot(p1, p2, p3, p4, layout=(2, 2))

savefig("synchronized_chaos_lidar.png")

display(plt)