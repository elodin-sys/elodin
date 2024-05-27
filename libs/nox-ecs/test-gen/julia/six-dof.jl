using Quaternionic
using DifferentialEquations

Ω⃗ = quatvec(0, 1, 0, 0)
ω⃗(t) = Ω⃗
q₀ = rotor(1, 0, 0, 0)

q_exact = @. exp([Ω⃗] * 1.0 / 2) * q₀
println(q_exact)

Ω⃗ = quatvec(0, 0, 1, 0)
ω⃗(t) = Ω⃗
q₀ = rotor(1, 0, 0, 0)

q_exact = @. exp([Ω⃗] * 1.0 / 2) * q₀
println(q_exact)

Ω⃗ = quatvec(0, 1, 1, 0)
ω⃗(t) = Ω⃗
q₀ = rotor(1, 0, 0, 0)

q_exact = @. exp([Ω⃗] * 1.0 / 2) * q₀
println(q_exact)
