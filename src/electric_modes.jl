besselj2(x) = besselj(2, x)
besselk2(x) = besselk(2, x)
besselj1_derivative(x) = 1 / 2 * (besselj0(x) - besselj2(x))
besselk1_derivative(x) = -1 / 2 * (besselk0(x) + besselk2(x))
besselj_derivative(m, x) = 1 / 2 * (besselj(m - 1, x) - besselj(m + 1, x))
besselk_derivative(m, x) = -1 / 2 * (besselk(m - 1, x) + besselk(m + 1, x))
hankelh1_derivative(m, x) = 1 / 2 * (hankelh1(m - 1, x) - hankelh1(m + 1, x))
hankelh2_derivative(m, x) = 1 / 2 * (hankelh2(m - 1, x) - hankelh2(m + 1, x))

"""
    electric_guided_mode_cylindrical_base_components(ρ::Real, a::Real, β::Real, p::Real, q::Real, K1J1::Real, s::Real)

Compute the underlying cylindrical components of the guided mode electric field used in the
expressions for both the quasilinear and quasicircular guided modes.

These components for ``\\rho < a`` are given by
```math
\\begin{aligned}
    e_{\\rho} &= \\mathrm{i} \\frac{q}{p} \\frac{K_{1}(q a)}{J_{1}(p a)} [(1 - s) J_{0}(p \\rho) - (1 + s) J_{2}(p \\rho)] \\\\
    e_{\\phi} &= -\\frac{q}{p} \\frac{K_{1}(q a)}{J_{1}(p a)} [(1 - s) J_{0}(p \\rho) + (1 + s) J_{2}(p \\rho)] \\\\
    e_{z} &= \\frac{2 q}{\\beta} \\frac{K_{1}(q a)}{J_{1}(p a)} J_{1}(p \\rho),
\\end{aligned}
```
and the components for ``\\rho > a`` are given by
```math
\\begin{aligned}
    e_{\\rho} &= \\mathrm{i} [(1 - s) K_{0}(q \\rho) + (1 + s) K_{2}(q \\rho)] \\\\
    e_{\\phi} &= -[(1 - s) K_{0}(q \\rho) - (1 + s) K_{2}(q \\rho)] \\\\
    e_{z} &= \\frac{2 q}{\\beta} K_{1}(q \\rho),
\\end{aligned}
```
where ``a`` is the fiber radius, ``\\beta`` is the propagation constant,
``p = \\sqrt{n^2 k^2 - \\beta^2}``, and ``q = \\sqrt{\\beta^2 - k^2}``, with ``k`` being the
free space wavenumber of the light. Futhermore, ``J_n`` and ``K_n`` are Bessel functions of
the first kind, and modified Bessel functions of the second kind, respectively, and the
prime denotes the derivative. Lastly, ``s`` is defined as
```math
s = \\frac{\\frac{1}{p^2 a^2} + \\frac{1}{q^2 a^2}}{\\frac{J_{1}'(p a)}{p a J_{1}(p a)} + \\frac{K_{1}'(q a)}{q a K_{1}(q a)}}.
```
"""
function electric_guided_mode_cylindrical_base_components(ρ::Real, fiber::Fiber)
    a = radius(fiber)
    β = propagation_constant(fiber)
    h = fiber.internal_parameter
    q = fiber.external_parameter
    K1J1 = fiber.besselk1_over_besselj1
    s = fiber.s_parameter

    if ρ < a
        e_ρ = im * q / h * K1J1 * ((1 - s) * besselj0(h * ρ) - (1 + s) * besselj2(h * ρ))
        e_ϕ = -q / h * K1J1 * ((1 - s) * besselj0(h * ρ) + (1 + s) * besselj2(h * ρ))
        e_z = 2 * q / β * K1J1 * besselj1(h * ρ)
    else
        e_ρ = im * ((1 - s) * besselk0(q * ρ) + (1 + s) * besselk2(q * ρ))
        e_ϕ = -((1 - s) * besselk0(q * ρ) - (1 + s) * besselk2(q * ρ))
        e_z = 2 * q / β * besselk1(q * ρ)
    end

    return e_ρ, e_ϕ, e_z
end

"""
    electric_guided_mode_profile_cartesian_components(ρ::Real, ϕ::Real, l::Integer, f::Integer, fiber::Fiber, ::CircularPolarization)

Compute the cartesian components of the mode profile of the guided electric field in the
plane transverse to the fiber.

The components are given by
```math
\\begin{aligned}
    e_{x} &= A (e_{\\rho} \\cos(\\phi) - l e_{\\phi} \\sin(\\phi)) \\mathrm{e}^{\\mathrm{i} l \\phi} \\\\
    e_{y} &= A (e_{\\rho} \\sin(\\phi) + l e_{\\phi} \\cos(\\phi)) \\mathrm{e}^{\\mathrm{i} l \\phi} \\\\
    e_{z} &= f A e_{z} \\mathrm{e}^{\\mathrm{i} l \\phi},
\\end{aligned}
```
where ``A`` is the normalization constant, ``l`` is the polarization, and ``f`` is the
direction of propagation. The base components, ``e_{\\rho}``, ``e_{\\phi}``, and
``e_{\\rho}``, are given by [`electric_guided_mode_cylindrical_base_components`](@ref).
"""
function electric_guided_mode_profile_cartesian_components(ρ::Real, ϕ::Real, l::Integer, f::Integer, fiber::Fiber, ::CircularPolarization)
    l ⊆ (-1, 1) || throw(DomainError(l, "Polarization number must be either +1 or -1."))
    f ⊆ (-1, 1) || throw(DomainError(f, "Direction of propagation must be either +1 or -1."))

    C = fiber.normalization_constant

    e_ρ, e_ϕ, e_z = electric_guided_mode_cylindrical_base_components(ρ, fiber)
    e_x = C * (e_ρ * cos(ϕ) - l * e_ϕ * sin(ϕ)) * exp(im * l * ϕ)
    e_y = C * (e_ρ * sin(ϕ) + l * e_ϕ * cos(ϕ)) * exp(im * l * ϕ)
    e_z = C * f * e_z * exp(im * l * ϕ)

    return e_x, e_y, e_z
end

"""
    electric_guided_field_cartesian_components(ρ::Real, ϕ::Real, l::Integer, f::Integer, fiber::Fiber, polarization_basis::CircularPolarization, power::Real)

Compute the cartesian components of a guided electric field at position ``(ρ, ϕ, z)`` and
time ``t`` with the given ``power``.
"""
function electric_guided_field_cartesian_components(ρ::Real, ϕ::Real, z::Real, t::Real, l::Integer, f::Integer, fiber::Fiber, polarization_basis::CircularPolarization, power::Real)
    ω = frequency(fiber)
    β = propagation_constant(fiber)
    e_x, e_y, e_z = electric_guided_mode_profile_cartesian_components(ρ, ϕ, l, f, fiber, polarization_basis)
    power_and_phase_factor = sqrt(power * propagation_constant_derivative(fiber) / 2) * exp(im * (f * β * z - ω * t))
    return e_x * power_and_phase_factor, e_y * power_and_phase_factor, e_z * power_and_phase_factor
end

"""
    electric_guided_field_cartesian_vector(ρ::Real, ϕ::Real, l::Integer, f::Integer, fiber::Fiber, polarization_basis::CircularPolarization, power::Real)

Compute the guided electric field vector at position ``(ρ, ϕ, z)`` and time ``t`` with the
given ``power``.
"""
function electric_guided_field_cartesian_vector(ρ::Real, ϕ::Real, l::Integer, f::Integer, fiber::Fiber, polarization_basis::CircularPolarization, power::Real)
    return collect(electric_guided_field_cartesian_components(ρ, ϕ, z, t, l, f, fiber, polarization_basis, power))
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

function electric_radiation_mode_cylindrical_components(ρ, a, n, ω, l, m, β, polarization_basis::CircularPolarization)
    if ρ < a
        return electric_radiation_mode_cylindrical_components_internal(ρ, a, n, ω, l, m, β, polarization_basis)
    else
        return electric_radiation_mode_cylindrical_components_external(ρ, a, n, ω, l, m, β, polarization_basis)
    end
end

function electric_radiation_mode(ρ, ϕ, ω, l, m, β, fiber::Fiber{T}, ploarization_basis::CircularPolarization) where {T<:Number}
    a = fiber.radius
    n = fiber.refractive_index
    e_ρ, e_ϕ, e_z = electric_radiation_mode_cylindrical_components(ρ, a, n, ω, l, m, β, ploarization_basis)
    return [e_ρ * cos(ϕ) - e_ϕ * sin(ϕ), e_ρ * sin(ϕ) + e_ϕ * cos(ϕ), e_z]
end
