# 2022/03/22
# 论文：Nonlinear Dynamics of Interband Cascade Laser Subjected to Optical Feedback
# 参考论文和代码见：参考资料/带间级联激光器

using DifferentialEquations

const ⋅ = *
function rate_equation(du, u, h, p, t)
  I, ω, τ_f, η, q, Γₚ, v_g, a₀, Nₜᵣ, τₛₚ, k, m, τ_aug, αₕ, τₚ, β, A = p
  N, S, ϕ = u
  N = N > 0 ? N : 0
  S = S > 0 ? S : 1
  _, S_, ϕ_ = h(p, t - τ_f)

  g = a₀ ⋅ (N - Nₜᵣ) / A
  θ = ω * τ_f + ϕ - ϕ_

  Ṅ = η ⋅ I / q - Γₚ ⋅ v_g ⋅ g ⋅ S - N / τₛₚ - N / τ_aug
  Ṡ = (m ⋅ Γₚ ⋅ v_g ⋅ g - 1 / τₚ)S + m ⋅ β ⋅ N / τₛₚ + 2k ⋅ √(S ⋅ S_) ⋅ cos(θ)
  ϕ̇ = αₕ / 2 ⋅ (m ⋅ Γₚ ⋅ v_g ⋅ g - 1 / τₚ) - k ⋅ √(S_ / S) ⋅ sin(θ)

  du[1] = Ṅ
  du[2] = Ṡ
  du[3] = ϕ̇
end

η = 0.64 # 注入电流效率
q = 1.6e-19 # 电子电荷(c)
c = 3e8 # 光速(m/s)
λ = 3.7e-6 # 波长(m)
L = 2e-3 # 腔长(m)
W = 4.4e-6 # 腔宽(m)
nᵣ = 3.58 # 折射率
m = 5 # 量子级联数
Γₚ = 0.04 # 光限制因子
v_g = 8.38e7 #  群速度(m/s)
a₀ = 2.8e-12 #  增益
Nₜᵣ = 6.2e7#   阈值载流子数
A = 8.8e-9#  有源区面积(m^2)
τₛₚ = 15e-9#  自发辐射寿命(s)
τ_aug = 1.08e-9#  Auger寿命(s)
τₚ = 10.5e-12#  光子寿命(s)
τᵢₙ = 2 * L / c * nᵣ#  腔内往返时间(s)
τ_f = 1.44e-9 #  光反馈延迟时间(s)
ω = 2π * c / λ #  光的角频率
β = 1e-4#  自发发射因子（？）
R = 0.32# 反射镜反射率
Iₜₕ = q / H * (1 / m * A / (Γₚ * v_g * a₀ * τₚ) + Nₜᵣ) * (1 / τₛₚ + 1 / τ_aug)#  阈值电流
I = 1.01Iₜₕ #          注入电流
Cₗ = (1 - R) / (2 * √R)#  外部耦合系数
f_ext_array=[0.0 3 0.14 0.2 0.84]*0.01 # 反馈光强和激光器输出光强比值区间
f_ext=f_ext_array[2]  # 反馈光强和激光器输出光强比值
k = 2Cₗ⋅√f_ext/τᵢₙ
αₕ = 2.2#  线宽增强因子

# Solver parameters
tspan = (0.0, 600e-9)
u0 = [0.5Nₜᵣ, 1, 0]
h(p, t) = u0
lags = [τ_f]

prob = DDEProblem(rate_equation, u0, h, tspan, [I, ω, τ_f, η, q, Γₚ, v_g, a₀, Nₜᵣ, τₛₚ, k, m, τ_aug, αₕ, τₚ, β, A]; constant_lags = lags)
sol = solve(prob; alg_hints = [:auto], reltol = 1e-8, abstol = 1, dt = 0.1e-12, saveat = 0.1e-9)

using Plots
using FFTW
p1 = plot(sol; vars = [2], label = "\$S(t)\$")
p2 = plot(sol; vars = [1], label = "\$N(t)\$")

fs = (sol.t[end] - sol.t[1]) / (length(sol.t) - 1)
freqs = fftfreq(length(sol.t), fs)
freqs = fftshift(freqs)

S = [u[2] for u in sol.u]
S = fftshift(fft(S))

p3 = plot(freqs, 20log10.(abs.(S)), label="\$\\hat{S}\$")

plot(p3)
