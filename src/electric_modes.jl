besselj2(x) = besselj(2, x)
besselk2(x) = besselk(2, x)
besselj1_derivative(x) = 1 / 2 * (besselj0(x) - besselj2(x))
besselk1_derivative(x) = -1 / 2 * (besselk0(x) + besselk2(x))
besselj_derivative(m, x) = 1 / 2 * (besselj(m - 1, x) - besselj(m + 1, x))
besselk_derivative(m, x) = -1 / 2 * (besselk(m - 1, x) + besselk(m + 1, x))
hankelh1_derivative(m, x) = 1 / 2 * (hankelh1(m - 1, x) - hankelh1(m + 1, x))
hankelh2_derivative(m, x) = 1 / 2 * (hankelh2(m - 1, x) - hankelh2(m + 1, x))

abstract type PolarizationBasis end
struct LinearPolarization <: PolarizationBasis end
struct CircularPolarization <: PolarizationBasis end


#TODO Implement the different variations of
# Guided mode or radiation mode
# Circular polarization or linear polarization
# Cartesian or cylindrical components
# Componenet returned individually or in a vector
#
# This is 2^4 = 16 different implementations

struct Fiber{T<:Real}
    radius::T
    wavelength::T
    frequency::T
    material::Material{T}
    refractive_index::T
    propagation_constant::T
    propagation_constant_derivative::T
    normalized_frequency::T
    internal_parameter::T
    external_parameter::T
    besselj1::T
    besselk1::T
    s::T
    normalization_constant::T
    function Fiber(radius::T, wavelength::T, material::Material{T}) where {T<:Real}
        ω = 2π / wavelength
        n = sellmeier_equation(material, wavelength)
        β = propagation_constant(radius, n, ω)
        dβ = propagation_constant_derivative(radius, n, ω)
        V = normalized_frequency(radius, n, ω)
        V > 2.40482 && @warn "Fiber supports multiple modes."
        p = sqrt(n^2 * ω^2 - β^2)
        q = sqrt(β^2 - ω^2)
        pa = p * radius
        qa = q * radius
        J1 = besselj1(pa)
        dJ1 = besselj1_derivative(pa)
        K1 = besselk1(qa)
        dK1 = besselk1_derivative(qa)
        s = (1 / pa^2 + 1 / qa^2) / (dJ1 / (pa * J1) + dK1 / (qa * K1))
        C = electric_mode_normalization_constant(radius, n, β, p, q, J1, K1, s)

        return new{T}(radius, wavelength, ω, material, n, β, dβ, V, p, q, J1, K1, s, C)
    end
end

function Base.show(io::IO, fiber::Fiber{T}) where {T<:Real}
    println(io, "Optical fiber with parameters:")
    println(io, "Radius: $(fiber.radius)μm")
    println(io, "Wavelength: $(fiber.wavelength)μm")
    println(io, "Frequency: 2π • $(fiber.frequency / (2π))")
    println(io, "Refractive index: $(fiber.refractive_index)")
    println(io, "Propagation constant: $(fiber.propagation_constant)")
    print(io, fiber.material)
end

function normalized_frequency(a, n, ω)
    return ω * a * sqrt(n^2 - 1)
end

function effective_refractive_index(λ, β)
    return β * λ / 2π
end


"""
    electric_guided_mode_cylindrical_components(ρ::Real, a::Real, β::Real, p::Real, q::Real, J1::Real, K1::Real, s::Real)

Compute the underlying cylindrical components of the guided mode electric field used in the
expressions for both the quasilinear and quasicircular guided modes.

These components for ``\\rho < a`` are given by

```math
\\begin{aligned}
e_{\\rho} &= \\mathrm{i} \\frac{q}{p} \\frac{K_{1}(q a)}{J_{1}(p a)} [(1 - s) J_{0}(p \\rho) - (1 + s) J_{2}(p \\rho)] \\\\
e_{\\phi} &= -\\frac{q}{p} \\frac{K_{1}(q a)}{J_{1}(p a)} [(1 - s) J_{0}(p \\rho) + (1 + s) J_{2}(p \\rho)] \\\\
e_{z} &= \\frac{2 q}{\\beta} \\frac{K_{1}(q a)}{J_{1}(p a)} J_{1}(p \\rho)
\\end{aligned}
```
"""
function electric_guided_mode_cylindrical_components(ρ::Real, a::Real, β::Real, p::Real, q::Real, J1::Real, K1::Real, s::Real)
    if ρ < a
        e_ρ = im * q / p * K1 / J1 * ((1 - s) * besselj0(p * ρ) - (1 + s) * besselj2(p * ρ))
        e_ϕ = -q / p * K1 / J1 * ((1 - s) * besselj0(p * ρ) + (1 + s) * besselj2(p * ρ))
        e_z = 2 * q / β * K1 / J1 * besselj1(p * ρ)
    else
        e_ρ = im * ((1 - s) * besselk0(q * ρ) + (1 + s) * besselk2(q * ρ))
        e_ϕ = -((1 - s) * besselk0(q * ρ) - (1 + s) * besselk2(q * ρ))
        e_z = 2 * q / β * besselk1(q * ρ)
    end

    return e_ρ, e_ϕ, e_z
end

function electric_mode_components_outside_cartesian(ρ, ϕ, l::Integer, f::Integer, fiber::Fiber{T}, ::CircularPolarization) where {T<:Number}
    β = fiber.propagation_constant
    q = fiber.external_parameter
    s = fiber.s
    C = fiber.normalization_constant
    e_ρ = im * ((1 - s) * besselk0(q * ρ) + (1 + s) * besselk2(q * ρ))
    e_ϕ = -((1 - s) * besselk0(q * ρ) - (1 + s) * besselk2(q * ρ))
    e_z = 2 * q / β * besselk1(q * ρ)
    e_x = C * (e_ρ * cos(ϕ) - l * e_ϕ * sin(ϕ))
    e_y = C * (e_ρ * sin(ϕ) + l * e_ϕ * cos(ϕ))
    e_z = C * f * e_z
    return e_x, e_y, e_z
end

function electric_mode_normalization_constant_integrand(ρ, parameters)
    a, n, β, p, q, J1, K1, s = parameters
    e_ρ, e_ϕ, e_z = electric_mode_components(ρ, a, β, p, q, J1, K1, s)
    abs2e = abs2(e_ρ) + abs2(e_ϕ) + abs2(e_z)
    if ρ < a
        return ρ * n^2 * abs2e
    else
        return ρ * abs2e
    end
end

function electric_mode_normalization_constant(a, n, β, p, q, J1, K1, s)
    parameters = (a, n, β, p, q, J1, K1, s)
    domain = (0.0, Inf)
    problem = IntegralProblem(electric_mode_normalization_constant_integrand, domain, parameters)
    solution = solve(problem, HCubatureJL())
    return 1 / sqrt(2π * solution.u)
end

function electric_mode(ρ::Real, ϕ::Real, l::Integer, f::Integer, fiber::Fiber{T}, ::LinearPolarization) where {T<:Number}
    a = fiber.radius
    β = fiber.propagation_constant
    p = fiber.internal_parameter
    q = fiber.external_parameter
    J1 = fiber.besselj1
    K1 = fiber.besselk1
    s = fiber.s
    C = fiber.normalization_constant

    e_ρ, e_ϕ, e_z = electric_mode_components(ρ, a, β, p, q, J1, K1, s)

    return sqrt(2) * C * [e_ρ * cos(ϕ) * cos(ϕ - ϕ₀) - im * e_ϕ * sin(ϕ) * sin(ϕ - ϕ₀), e_ρ * sin(ϕ) * cos(ϕ - ϕ₀) + im * e_ϕ * cos(ϕ) * sin(ϕ - ϕ₀), f * e_z * cos(ϕ - ϕ₀)]
end

function electric_guided_mode_cartesian(ρ::Real, ϕ::Real, l::Integer, f::Integer, fiber::Fiber{T}, ::CircularPolarization) where {T<:Number}
    a = fiber.radius
    β = fiber.propagation_constant
    p = fiber.internal_parameter
    q = fiber.external_parameter
    J1 = fiber.besselj1
    K1 = fiber.besselk1
    s = fiber.s
    C = fiber.normalization_constant

    e_ρ, e_ϕ, e_z = electric_mode_components(ρ, a, β, p, q, J1, K1, s)

    return C * [e_ρ * cos(ϕ) - l * e_ϕ * sin(ϕ), e_ρ * sin(ϕ) + l * e_ϕ * cos(ϕ), f * e_z]
end

function electric_radiation_mode_cylindrical_components_internal(ρ, a, n, ω, l, m, β, ::CircularPolarization)
    p = sqrt(ω^2 * n^2 - β^2)
    q = sqrt(ω^2 - β^2)
    V_1 = m * ω * β / (a * p^2 * q^2) * (1 - n^2) * besselj(m, p * a) * conj(hankelh1(m, q * a))
    M_1 = 1 / p * besselj_derivative(m, p * a) * conj(hankelh1(m, q * a)) - 1 / q * besselj(m, p * a) * conj(hankelh1_derivative(m, q * a))
    L_1 = n^2 / p * besselj_derivative(m, p * a)* conj(hankelh1(m, q * a)) - 1 / q * besselj(m, p * a) * conj(hankelh1_derivative(m, q * a))
    η = sqrt((abs2(V_1) + abs2(L_1)) / (abs2(V_1) + abs2(M_1)))
    B = im * l * η
    C_1 = -im * π * q^2 * a / 4 * (L_1 + im * B * V_1)
    D_1 = im * π * q^2 * a / 4 * (im * V_1 - B * M_1)
    N_ν = 8π * ω / (q^2) * (abs2(C_1) + abs2(D_1))
    A = 1 / sqrt(N_ν)
    e_ρ = A * im / (p^2) * (β * p * besselj_derivative(m, p * ρ) + im * m * ω / ρ * B * besselj(m, p * ρ))
    e_ϕ = A * im / (p^2) * (im * m * β / ρ * besselj(m, p * ρ) - p * ω * B * besselj_derivative(m, p * ρ))
    e_z = A * besselj(m, p * ρ)

    return e_ρ, e_ϕ, e_z
end

function electric_radiation_mode_cylindrical_components_external(ρ, a, n, ω, l, m, β, ::CircularPolarization)
    p = sqrt(ω^2 * n^2 - β^2)
    q = sqrt(ω^2 - β^2)
    V_1 = m * ω * β / (a * p^2 * q^2) * (1 - n^2) * besselj(m, p * a) * conj(hankelh1(m, q * a))
    V_2 = m * ω * β / (a * p^2 * q^2) * (1 - n^2) * besselj(m, p * a) * conj(hankelh2(m, q * a))
    M_1 = 1 / p * besselj_derivative(m, p * a) * conj(hankelh1(m, q * a)) - 1 / q * besselj(m, p * a) * conj(hankelh1_derivative(m, q * a))
    M_2 = 1 / p * besselj_derivative(m, p * a) * conj(hankelh2(m, q * a)) - 1 / q * besselj(m, p * a) * conj(hankelh2_derivative(m, q * a))
    L_1 = n^2 / p * besselj_derivative(m, p * a)* conj(hankelh1(m, q * a)) - 1 / q * besselj(m, p * a) * conj(hankelh1_derivative(m, q * a))
    L_2 = n^2 / p * besselj_derivative(m, p * a)* conj(hankelh2(m, q * a)) - 1 / q * besselj(m, p * a) * conj(hankelh2_derivative(m, q * a))
    η = sqrt((abs2(V_1) + abs2(L_1)) / (abs2(V_1) + abs2(M_1)))
    B = im * l * η
    C_1 = -im * π * q^2 * a / 4 * (L_1 + im * B * V_1)
    C_2 = im * π * q^2 * a / 4 * (L_2 + im * B * V_2)
    D_1 = im * π * q^2 * a / 4 * (im * V_1 - B * M_1)
    D_2 = -im * π * q^2 * a / 4 * (im * V_2 - B * M_2) 
    N_ν = 8π * ω / (q^2) * (abs2(C_1) + abs2(D_1))
    A = 1 / sqrt(N_ν)

    e_ρ = A * im / (q^2) * (β * q * C_1 * hankelh1_derivative(m, q * ρ) + im * m * ω / ρ * D_1 * hankelh1(m, q * ρ) + β * q * C_2 * hankelh2_derivative(m, q * ρ) + im * m * ω / ρ * D_2 * hankelh2(m, q * ρ))
    e_ϕ = A * im / (q^2) * (im * m * β / ρ * C_1 * hankelh1(m, q * ρ) - q * ω * D_1 * hankelh1_derivative(m, q * ρ) + im * m * β / ρ * C_2 * hankelh2(m, q * ρ) - q * ω * D_2 * hankelh2_derivative(m, q * ρ))
    e_z = A * (C_1 * hankelh1(m, q * ρ) + C_2 * hankelh2(m, q * ρ))

    return e_ρ, e_ϕ, e_z
end

function electric_radiation_mode_cylindrical_components(ρ, a, n, ω, l, m, β, ploarization_basis::CircularPolarization)
    if ρ < a
        return electric_radiation_mode_cylindrical_components_internal(ρ, a, n, ω, l, m, β, ploarization_basis)
    else
        return electric_radiation_mode_cylindrical_components_external(ρ, a, n, ω, l, m, β, ploarization_basis)
    end
end

function electric_radiation_mode(ρ, ϕ, ω, l, m, β, fiber::Fiber{T}, ploarization_basis::CircularPolarization) where {T<:Number}
    a = fiber.radius
    n = fiber.refractive_index
    e_ρ, e_ϕ, e_z = electric_radiation_mode_cylindrical_components(ρ, a, n, ω, l, m, β, ploarization_basis)
    return [e_ρ * cos(ϕ) - e_ϕ * sin(ϕ), e_ρ * sin(ϕ) + e_ϕ * cos(ϕ), e_z]
end
