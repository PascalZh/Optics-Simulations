# 2022/03/09
# 功能：解出agraval的semiconductor lasers一书中第六章给出的速率方程；画出Figure6.1以及Figure6.2
# 参考：./参考资料/单模FP腔自由运行
# 速率方程：单模、FP腔 

using DifferentialEquations

const ⋅ = *
function rate_equation1(du, u, p, t)
  I, γ, q, V, n₀, Γ, v_g, a, βₛₚ, Aₙᵣ, B, C = p
  P, N = u

  n = N/V

  G = Γ⋅v_g⋅a⋅(n - n₀)
  Rₛₚ = βₛₚ⋅B⋅N^2/V
  γₑ = Aₙᵣ + B⋅n + C⋅n^2

  Ṗ = (G - γ)P + Rₛₚ
  Ṅ = I/q - γₑ⋅N - G⋅P

  du[1] = Ṗ
  du[2] = Ṅ
end

Aₙᵣ = 1e8
B = 1e-16
C = 3e-41

a = 2.5e-20
τₚ = 1.6e-12
τₑ = 2.2e-9
γ = 1/τₚ
γₑ = 1/τₑ

αₘ = 45e2
αᵢₙₜ = 40e2

q = 1.6e-19

L = 250e-6
w = 2e-6
d = 0.2e-6
V = L⋅w⋅d # 腔的体积
n₀ = 1e24

Γ = 0.3

v_g = γ/(αₘ + αᵢₙₜ)

βₛₚ = 1e-3

# Solver parameters
tspan = (0.0, 1e-7)

using Plots
# pyplot()

# {{{
# anim = @animate for I = (0:0.23:30) * 1e-3
#   prob = ODEProblem(rate_equation1, [0, 1e-3], tspan, [I, γ, q, V, n₀, Γ, v_g, a, βₛₚ, Aₙᵣ, B, C])
#   sol = solve(prob, Tsit5(); reltol = 1e-8, abstol = 1, dt = tspan[2] / 1000, saveat = tspan[2] / 1000)
#   p1 = plot(sol, xaxis = "Time (t)", yaxis = "P(t)", label = "Photon number", vars = [1])
#   ylims!(0, 6e5)
#   p2 = plot(sol, xaxis = "Time (t)", yaxis = "N(t)", label = "Carrier number", vars = [2], color = :green)
#   ylims!(0, 3e8)
#   plot(p1, p2, layout = (1, 2))
#   title!("I(current) = $(trunc(Int, I*1e3))mA")
# end

# gif(anim, "rate equation with I(current) varying.gif")
# }}}

Is = (0:0.01:30) * 1e-3
Ps = Vector(undef, length(Is))
Ns = Vector(undef, length(Is))
@time Threads.@threads for i ∈ 1:length(Is)
  prob = ODEProblem(rate_equation1, [0, 1e3], tspan, [Is[i], γ, q, V, n₀, Γ, v_g, a, βₛₚ, Aₙᵣ, B, C])
  sol = solve(prob; alg_hints = [:auto], reltol = 1e-8, abstol = 1, dt = tspan[2] / 1000, saveat = tspan[2] / 1000)
  len = length(sol.t)
  steady_u = sum(sol.u[len*9÷10:end]) / 0.1len
  Ps[i] = steady_u[1]
  Ns[i] = steady_u[2]
end

# Figure 6.1
p1 = plot(1e3Is, @. log10(abs(Ps)))
p2 = plot(1e3Is, Ns)
plot(p1, p2)