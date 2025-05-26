abstract type Polarization end

struct LinearPolarization <: Polarization
    ϕ₀::Float64
end

LinearPolarization(ϕ₀::Real) = LinearPolarization(Float64(ϕ₀))

function LinearPolarization(axis::Char)
    if axis ∈ ('X', 'x')
        return LinearPolarization(0)
    elseif axis ∈ ('Y', 'y')
        return LinearPolarization(π / 2)
    else
        error("Expected polarization axis to be either 'x' or 'y', got: '$axis'.")
    end
end

LinearPolarization(axis::String) = LinearPolarization(only(axis))
LinearPolarization(sym::Symbol) = LinearPolarization(string(sym))

struct CircularPolarization <: Polarization
    l::Int
    function CircularPolarization(l::Int)
        l ∈ (-1, 1) || throw(DomainError(l, "Polarization index must be either +1 or -1."))
        return new(l)
    end
end

"""
    electric_guided_mode_cylindrical_base_components(ρ::Real, a::Real, β::Real, p::Real, q::Real, K1J1::Real, s::Real)

Compute the underlying cylindrical components of the guided mode electric field used in the
expressions for both the quasilinear and quasicircular guided modes.

These components for ``\\rho < a`` are given by
```math
\\begin{aligned}
    e_{\\rho} &= A \\mathrm{i} \\frac{q}{p} \\frac{K_{1}(q a)}{J_{1}(p a)} [(1 - s) J_{0}(p \\rho) - (1 + s) J_{2}(p \\rho)] \\\\
    e_{\\phi} &= -A \\frac{q}{p} \\frac{K_{1}(q a)}{J_{1}(p a)} [(1 - s) J_{0}(p \\rho) + (1 + s) J_{2}(p \\rho)] \\\\
    e_{z} &= A \\frac{2 q}{\\beta} \\frac{K_{1}(q a)}{J_{1}(p a)} J_{1}(p \\rho),
\\end{aligned}
```
and the components for ``\\rho > a`` are given by
```math
\\begin{aligned}
    e_{\\rho} &= A \\mathrm{i} [(1 - s) K_{0}(q \\rho) + (1 + s) K_{2}(q \\rho)] \\\\
    e_{\\phi} &= -A [(1 - s) K_{0}(q \\rho) - (1 + s) K_{2}(q \\rho)] \\\\
    e_{z} &= A \\frac{2 q}{\\beta} K_{1}(q \\rho),
\\end{aligned}
```
where ``A`` is the normalization constant, ``a`` is the fiber radius, ``\\beta`` is the
propagation constant, ``p = \\sqrt{n^2 k^2 - \\beta^2}``, and
``q = \\sqrt{\\beta^2 - k^2}``, with ``k`` being the free space wavenumber of the light.
Futhermore, ``J_n`` and ``K_n`` are Bessel functions of the first kind, and modified Bessel
functions of the second kind, respectively, and the prime denotes the derivative. Lastly,
``s`` is defined as
```math
s = \\frac{\\frac{1}{p^2 a^2} + \\frac{1}{q^2 a^2}}{\\frac{J_{1}'(p a)}{p a J_{1}(p a)} + \\frac{K_{1}'(q a)}{q a K_{1}(q a)}}.
```
"""
function electric_guided_mode_cylindrical_base_components(ρ::Real, fiber::Fiber)
    ρ ≥ 0.0 || throw(DomainError(ρ, "Radial coordinate must be non-negative."))

    a = radius(fiber)
    β = propagation_constant(fiber)
    h = fiber.internal_parameter
    q = fiber.external_parameter
    s = fiber.s_parameter
    A = fiber.normalization_constant

    if ρ < a
        K1J1 = besselk1(q * a) / besselj1(h * a)
        e_ρ = im * A * q / h * K1J1 * ((1 - s) * besselj0(h * ρ) - (1 + s) * besselj(2, h * ρ))
        e_ϕ = -A * q / h * K1J1 * ((1 - s) * besselj0(h * ρ) + (1 + s) * besselj(2, h * ρ))
        e_z = 2 * A * q / β * K1J1 * besselj1(h * ρ)
    else
        e_ρ = im * A * ((1 - s) * besselk0(q * ρ) + (1 + s) * besselk(2, q * ρ))
        e_ϕ = -A * ((1 - s) * besselk0(q * ρ) - (1 + s) * besselk(2, q * ρ))
        e_z = 2 * A * q / β * besselk1(q * ρ)
    end

    return e_ρ, e_ϕ, e_z
end

function electric_guided_mode_profile_cylindrical_components(ρ::Real, ϕ::Real, f::Integer, fiber::Fiber, polarization::LinearPolarization)
    f ∈ (-1, 1) || throw(DomainError(f, "Direction of propagation index must be either +1 or -1."))

    ϕ₀ = polarization.ϕ₀
    e_ρ, e_ϕ, e_z = electric_guided_mode_cylindrical_base_components(ρ, fiber)

    return sqrt(2) * e_ρ * cos(ϕ - ϕ₀), sqrt(2) * im * e_ϕ * sin(ϕ - ϕ₀), sqrt(2) * f * e_z * cos(ϕ - ϕ₀)
end

function electric_guided_mode_profile_cylindrical_components(ρ::Real, ϕ::Real, f::Integer, fiber::Fiber, polarization::CircularPolarization)
    f ∈ (-1, 1) || throw(DomainError(f, "Direction of propagation index must be either +1 or -1."))

    l = polarization.l
    e_ρ, e_ϕ, e_z = electric_guided_mode_cylindrical_base_components(ρ, fiber)

    return e_ρ * exp(im * l * ϕ), l * e_ϕ * exp(im * l * ϕ), f * e_z * exp(im * l * ϕ)
end

function electric_guided_mode_profile_cartesian_components(ρ::Real, ϕ::Real, f::Integer, fiber::Fiber, polarization::Polarization)
    e_ρ, e_ϕ, e_z = electric_guided_mode_profile_cylindrical_components(ρ, ϕ, f, fiber, polarization)

    e_x = e_ρ * cos(ϕ) - e_ϕ * sin(ϕ)
    e_y = e_ρ * sin(ϕ) + e_ϕ * cos(ϕ)

    return e_x, e_y, e_z
end

"""
    electric_guided_field_cartesian_components(ρ::Real, ϕ::Real, z::Real, t::Real, f::Integer, fiber::Fiber, polarization::Polarization, power::Real)

Compute the cartesian components of a guided electric field at position ``(ρ, ϕ, z)`` and
time ``t`` with the given ``power``.
"""
function electric_guided_field_cartesian_components(ρ::Real, ϕ::Real, z::Real, t::Real, f::Integer, fiber::Fiber, polarization::Polarization, power::Real)
    ω = frequency(fiber)
    β = propagation_constant(fiber)

    e_x, e_y, e_z = electric_guided_mode_profile_cartesian_components(ρ, ϕ, f, fiber, polarization)
    C = sqrt(power * propagation_constant_derivative(fiber) / 2)
    phase_factor = exp(im * (f * β * z - ω * t))

    return e_x * C * phase_factor, e_y * C * phase_factor, e_z * C * phase_factor
end

"""
    electric_guided_field_cartesian_vector(ρ::Real, ϕ::Real, l::Integer, f::Integer, fiber::Fiber, polarization_basis::CircularPolarization, power::Real)

Compute the guided electric field vector at position ``(ρ, ϕ, z)`` and time ``t`` with the
given ``power``.
"""
function electric_guided_field_cartesian_vector(ρ::Real, ϕ::Real, z::Real, t::Real, f::Integer, fiber::Fiber, polarization::Polarization, power::Real)
    return collect(electric_guided_field_cartesian_components(ρ, ϕ, z, t, f, fiber, polarization, power))
end

function electric_radiation_mode_cylindrical_base_components_internal(ρ, ω, β, l::Integer, m::Integer, fiber)
    a = fiber.radius
    n = fiber.refractive_index
    h = sqrt(n^2 * ω^2 - β^2)
    q = sqrt(ω^2 - β^2)

    Jma = besselj(m, h * a)
    dJma = 1 / 2 * (besselj(m - 1, h * a) - besselj(m + 1, h * a))
    h1ma = hankelh1(m, q * a)
    dh1ma = 1 / 2 * (hankelh1(m - 1, q * a) - hankelh1(m + 1, q * a))

    V_1 = m * ω * β / (a * h^2 * q^2) * (1 - n^2) * Jma * conj(h1ma)
    M_1 = 1 / h * dJma * conj(h1ma) - 1 / q * Jma * conj(dh1ma)
    L_1 = n^2 / h * dJma * conj(h1ma) - 1 / q * Jma * conj(dh1ma)
    η = sqrt((abs2(V_1) + abs2(L_1)) / (abs2(V_1) + abs2(M_1)))
    B = im * l * η
    C_1 = -im * π * q^2 * a / 4 * (L_1 + im * B * V_1)
    D_1 = im * π * q^2 * a / 4 * (im * V_1 - B * M_1)
    N_ν = 8π * ω / (q^2) * (abs2(C_1) + abs2(D_1))
    A = 1 / sqrt(N_ν)

    Jmρ = besselj(m, h * ρ)
    dJmρ = 1 / 2 * (besselj(m - 1, h * a) - besselj(m + 1, h * ρ))
    
    e_ρ = A * im / (h^2) * (β * h * dJmρ + im * m * ω / ρ * B * Jmρ)
    e_ϕ = A * im / (h^2) * (im * m * β / ρ * Jmρ - h * ω * B * dJmρ)
    e_z = A * Jmρ

    return e_ρ, e_ϕ, e_z
end

function electric_radiation_mode_cylindrical_base_components_external(ρ, ω, β, l::Integer, m::Integer, fiber)
    a = fiber.radius
    n = fiber.refractive_index
    h = sqrt(n^2 * ω^2 - β^2)
    q = sqrt(ω^2 - β^2)

    Jma = besselj(m, h * a)
    dJma = 1 / 2 * (besselj(m - 1, h * a) - besselj(m + 1, h * a))
    h1ma = hankelh1(m, q * a)
    h2ma = hankelh2(m, q * a)
    dh1ma = 1 / 2 * (hankelh1(m - 1, q * a) - hankelh1(m + 1, q * a))
    dh2ma = 1 / 2 * (hankelh2(m - 1, q * a) - hankelh2(m + 1, q * a))

    V_1 = m * ω * β / (a * h^2 * q^2) * (1 - n^2) * Jma * conj(h1ma)
    V_2 = m * ω * β / (a * h^2 * q^2) * (1 - n^2) * Jma * conj(h2ma)
    M_1 = 1 / h * dJma * conj(h1ma) - 1 / q * Jma * conj(dh1ma)
    M_2 = 1 / h * dJma * conj(h2ma) - 1 / q * Jma * conj(dh2ma)
    L_1 = n^2 / h * dJma * conj(h1ma) - 1 / q * Jma * conj(dh1ma)
    L_2 = n^2 / h * dJma * conj(h2ma) - 1 / q * Jma * conj(dh2ma)
    η = sqrt((abs2(V_1) + abs2(L_1)) / (abs2(V_1) + abs2(M_1)))
    B = im * l * η
    C_1 = -im * π * q^2 * a / 4 * (L_1 + im * B * V_1)
    C_2 = im * π * q^2 * a / 4 * (L_2 + im * B * V_2)
    D_1 = im * π * q^2 * a / 4 * (im * V_1 - B * M_1)
    D_2 = -im * π * q^2 * a / 4 * (im * V_2 - B * M_2) 
    N_ν = 8π * ω / (q^2) * (abs2(C_1) + abs2(D_1))
    A = 1 / sqrt(N_ν)

    h1mρ = hankelh1(m, q * ρ)
    h2mρ = hankelh2(m, q * ρ)
    dh1mρ = 1 / 2 * (hankelh1(m - 1, q * ρ) - hankelh1(m + 1, q * ρ))
    dh2mρ = 1 / 2 * (hankelh2(m - 1, q * ρ) - hankelh2(m + 1, q * ρ))

    e_ρ = A * im / (q^2) * (β * q * C_1 * dh1mρ
                            + im * m * ω / ρ * D_1 * h1mρ
                            + β * q * C_2 * dh2mρ
                            + im * m * ω / ρ * D_2 * h2mρ)
    e_ϕ = A * im / (q^2) * (im * m * β / ρ * C_1 * h1mρ
                            - q * ω * D_1 * dh1mρ
                            + im * m * β / ρ * C_2 * h2mρ
                            - q * ω * D_2 * dh2mρ)
    e_z = A * (C_1 * h1mρ + C_2 * h2mρ)

    return e_ρ, e_ϕ, e_z
end

function electric_radiation_mode_cylindrical_base_components_external_trigonometric(ρ, ω, θ, l::Integer, m::Integer, fiber)
    (θ < 0 || θ > π) && throw(DomainError(θ, "θ must be between 0 and π."))

    a = fiber.radius
    n = fiber.refractive_index
    β = ω * cos(θ)
    q = ω * sin(θ)
    h = sqrt(n^2 * ω^2 - β^2)

    Jma = besselj(m, h * a)
    dJma = 1 / 2 * (besselj(m - 1, h * a) - besselj(m + 1, h * a))
    h1ma = hankelh1(m, q * a)
    h2ma = hankelh2(m, q * a)
    dh1ma = 1 / 2 * (hankelh1(m - 1, q * a) - hankelh1(m + 1, q * a))
    dh2ma = 1 / 2 * (hankelh2(m - 1, q * a) - hankelh2(m + 1, q * a))

    V_1 = m * ω * β / (a * h^2 * q^2) * (1 - n^2) * Jma * conj(h1ma)
    V_2 = m * ω * β / (a * h^2 * q^2) * (1 - n^2) * Jma * conj(h2ma)
    M_1 = 1 / h * dJma * conj(h1ma) - 1 / q * Jma * conj(dh1ma)
    M_2 = 1 / h * dJma * conj(h2ma) - 1 / q * Jma * conj(dh2ma)
    L_1 = n^2 / h * dJma * conj(h1ma) - 1 / q * Jma * conj(dh1ma)
    L_2 = n^2 / h * dJma * conj(h2ma) - 1 / q * Jma * conj(dh2ma)
    η = sqrt((abs2(V_1) + abs2(L_1)) / (abs2(V_1) + abs2(M_1)))
    B = im * l * η
    C_1 = -im * π * q^2 * a / 4 * (L_1 + im * B * V_1)
    C_2 = im * π * q^2 * a / 4 * (L_2 + im * B * V_2)
    D_1 = im * π * q^2 * a / 4 * (im * V_1 - B * M_1)
    D_2 = -im * π * q^2 * a / 4 * (im * V_2 - B * M_2) 
    N_ν = 8π * ω / (q^2) * (abs2(C_1) + abs2(D_1))
    A = 1 / sqrt(N_ν)

    h1mρ = hankelh1(m, q * ρ)
    h2mρ = hankelh2(m, q * ρ)
    dh1mρ = 1 / 2 * (hankelh1(m - 1, q * ρ) - hankelh1(m + 1, q * ρ))
    dh2mρ = 1 / 2 * (hankelh2(m - 1, q * ρ) - hankelh2(m + 1, q * ρ))

    e_ρ = A * im / (q^2) * (β * q * C_1 * dh1mρ
                            + im * m * ω / ρ * D_1 * h1mρ
                            + β * q * C_2 * dh2mρ
                            + im * m * ω / ρ * D_2 * h2mρ)
    e_ϕ = A * im / (q^2) * (im * m * β / ρ * C_1 * h1mρ
                            - q * ω * D_1 * dh1mρ
                            + im * m * β / ρ * C_2 * h2mρ
                            - q * ω * D_2 * dh2mρ)
    e_z = A * (C_1 * h1mρ + C_2 * h2mρ)

    return e_ρ, e_ϕ, e_z
end

function electric_radiation_mode_cylindrical_base_components_external_coefficients(a, q, ω, β, m::Integer, fiber)
    n = fiber.refractive_index
    h = sqrt(n^2 * ω^2 - β^2)
    
    Jma = besselj(m, h * a)
    dJma = 0.5 * (besselj(m - 1, h * a) - besselj(m + 1, h * a))
    h1ma = hankelh1(m, q * a)
    h2ma = hankelh2(m, q * a)
    dh1ma = 0.5 * (hankelh1(m - 1, q * a) - hankelh1(m + 1, q * a))
    dh2ma = 0.5 * (hankelh2(m - 1, q * a) - hankelh2(m + 1, q * a))

    V_1 = m * ω * β / (a * h^2 * q^2) * (1 - n^2) * Jma * conj(h1ma)
    V_2 = m * ω * β / (a * h^2 * q^2) * (1 - n^2) * Jma * conj(h2ma)
    M_1 = 1 / h * dJma * conj(h1ma) - 1 / q * Jma * conj(dh1ma)
    M_2 = 1 / h * dJma * conj(h2ma) - 1 / q * Jma * conj(dh2ma)
    L_1 = n^2 / h * dJma * conj(h1ma) - 1 / q * Jma * conj(dh1ma)
    L_2 = n^2 / h * dJma * conj(h2ma) - 1 / q * Jma * conj(dh2ma)
    η = sqrt((abs2(V_1) + abs2(L_1)) / (abs2(V_1) + abs2(M_1)))

    return V_1, V_2, M_1, M_2, L_1, L_2, η
end

function electric_radiation_mode_cylindrical_base_components_external_polarization_coefficients(l, a, q, ω, V_1, V_2, M_1, M_2, L_1, L_2, η)
    B = im * l * η
    factor = im * π * q^2 * a / 4
    C_1 = -factor * (L_1 + im * B * V_1)
    C_2 = factor * (L_2 + im * B * V_2)
    D_1 = factor * (im * V_1 - B * M_1)
    D_2 = -factor * (im * V_2 - B * M_2) 
    N_ν = 8π * ω / (q^2) * (abs2(C_1) + abs2(D_1))
    A = 1 / sqrt(N_ν)

    return C_1, C_2, D_1, D_2, A
end

function electric_radiation_mode_cylindrical_base_components_external_hankel(m, q, ρ)
    h1mρ = hankelh1(m, q * ρ)
    h2mρ = hankelh2(m, q * ρ)
    dh1mρ = 0.5 * (hankelh1(m - 1, q * ρ) - hankelh1(m + 1, q * ρ))
    dh2mρ = 0.5 * (hankelh2(m - 1, q * ρ) - hankelh2(m + 1, q * ρ))

    return h1mρ, h2mρ, dh1mρ, dh2mρ
end

function _electric_radiation_mode_cylindrical_base_components_external(ρ, m, ω, β, q, C_1, C_2, D_1, D_2, A, h1mρ, h2mρ, dh1mρ, dh2mρ)
    e_ρ = A * im / (q^2) * (β * q * C_1 * dh1mρ
                            + im * m * ω / ρ * D_1 * h1mρ
                            + β * q * C_2 * dh2mρ
                            + im * m * ω / ρ * D_2 * h2mρ)
    e_ϕ = A * im / (q^2) * (im * m * β / ρ * C_1 * h1mρ
                            - q * ω * D_1 * dh1mρ
                            + im * m * β / ρ * C_2 * h2mρ
                            - q * ω * D_2 * dh2mρ)
    e_z = A * (C_1 * h1mρ + C_2 * h2mρ)

    return e_ρ, e_ϕ, e_z
end

function electric_radiation_mode_cylindrical_base_components_external_both_polarizations_two_atoms(ρ_i, ρ_j, ω, β, m::Integer, fiber)
    a = fiber.radius
    q = sqrt(ω^2 - β^2)

    V_1, V_2, M_1, M_2, L_1, L_2, η = electric_radiation_mode_cylindrical_base_components_external_coefficients(a, q, ω, β, m, fiber)
    
    h1mρ_i, h2mρ_i, dh1mρ_i, dh2mρ_i = electric_radiation_mode_cylindrical_base_components_external_hankel(m, q, ρ_i)
    h1mρ_j, h2mρ_j, dh1mρ_j, dh2mρ_j = electric_radiation_mode_cylindrical_base_components_external_hankel(m, q, ρ_j)
    
    C_1_plus, C_2_plus, D_1_plus, D_2_plus, A_plus = electric_radiation_mode_cylindrical_base_components_external_polarization_coefficients(1, a, q, ω, V_1, V_2, M_1, M_2, L_1, L_2, η)
    e_ρ_plus_i, e_ϕ_plus_i, e_z_plus_i = _electric_radiation_mode_cylindrical_base_components_external(ρ_i, m, ω, β, q, C_1_plus, C_2_plus, D_1_plus, D_2_plus, A_plus, h1mρ_i, h2mρ_i, dh1mρ_i, dh2mρ_i)
    e_ρ_plus_j, e_ϕ_plus_j, e_z_plus_j = _electric_radiation_mode_cylindrical_base_components_external(ρ_j, m, ω, β, q, C_1_plus, C_2_plus, D_1_plus, D_2_plus, A_plus, h1mρ_j, h2mρ_j, dh1mρ_j, dh2mρ_j)
    
    C_1_minus, C_2_minus, D_1_minus, D_2_minus, A_minus = electric_radiation_mode_cylindrical_base_components_external_polarization_coefficients(-1, a, q, ω, V_1, V_2, M_1, M_2, L_1, L_2, η)
    e_ρ_minus_i, e_ϕ_minus_i, e_z_minus_i = _electric_radiation_mode_cylindrical_base_components_external(ρ_i, m, ω, β, q, C_1_minus, C_2_minus, D_1_minus, D_2_minus, A_minus, h1mρ_i, h2mρ_i, dh1mρ_i, dh2mρ_i)
    e_ρ_minus_j, e_ϕ_minus_j, e_z_minus_j = _electric_radiation_mode_cylindrical_base_components_external(ρ_j, m, ω, β, q, C_1_minus, C_2_minus, D_1_minus, D_2_minus, A_minus, h1mρ_j, h2mρ_j, dh1mρ_j, dh2mρ_j)

    return e_ρ_plus_i, e_ϕ_plus_i, e_z_plus_i, e_ρ_minus_i, e_ϕ_minus_i, e_z_minus_i,
           e_ρ_plus_j, e_ϕ_plus_j, e_z_plus_j, e_ρ_minus_j, e_ϕ_minus_j, e_z_minus_j
end

function electric_radiation_mode_cylindrical_base_components_external_both_polarizations_two_atoms_trigonometric(ρ_i, ρ_j, ω, θ, m::Integer, fiber)
    (θ < 0 || θ > π) && throw(DomainError(θ, "θ must be between 0 and π."))

    a = fiber.radius
    β = ω * cos(θ)
    q = ω * sin(θ)
    
    V_1, V_2, M_1, M_2, L_1, L_2, η = electric_radiation_mode_cylindrical_base_components_external_coefficients(a, q, ω, β, m, fiber)
    
    h1mρ_i, h2mρ_i, dh1mρ_i, dh2mρ_i = electric_radiation_mode_cylindrical_base_components_external_hankel(m, q, ρ_i)
    h1mρ_j, h2mρ_j, dh1mρ_j, dh2mρ_j = electric_radiation_mode_cylindrical_base_components_external_hankel(m, q, ρ_j)
    
    C_1_plus, C_2_plus, D_1_plus, D_2_plus, A_plus = electric_radiation_mode_cylindrical_base_components_external_polarization_coefficients(1, a, q, ω, V_1, V_2, M_1, M_2, L_1, L_2, η)
    e_ρ_plus_i, e_ϕ_plus_i, e_z_plus_i = _electric_radiation_mode_cylindrical_base_components_external(ρ_i, m, ω, β, q, C_1_plus, C_2_plus, D_1_plus, D_2_plus, A_plus, h1mρ_i, h2mρ_i, dh1mρ_i, dh2mρ_i)
    e_ρ_plus_j, e_ϕ_plus_j, e_z_plus_j = _electric_radiation_mode_cylindrical_base_components_external(ρ_j, m, ω, β, q, C_1_plus, C_2_plus, D_1_plus, D_2_plus, A_plus, h1mρ_j, h2mρ_j, dh1mρ_j, dh2mρ_j)
    
    C_1_minus, C_2_minus, D_1_minus, D_2_minus, A_minus = electric_radiation_mode_cylindrical_base_components_external_polarization_coefficients(-1, a, q, ω, V_1, V_2, M_1, M_2, L_1, L_2, η)
    e_ρ_minus_i, e_ϕ_minus_i, e_z_minus_i = _electric_radiation_mode_cylindrical_base_components_external(ρ_i, m, ω, β, q, C_1_minus, C_2_minus, D_1_minus, D_2_minus, A_minus, h1mρ_i, h2mρ_i, dh1mρ_i, dh2mρ_i)
    e_ρ_minus_j, e_ϕ_minus_j, e_z_minus_j = _electric_radiation_mode_cylindrical_base_components_external(ρ_j, m, ω, β, q, C_1_minus, C_2_minus, D_1_minus, D_2_minus, A_minus, h1mρ_j, h2mρ_j, dh1mρ_j, dh2mρ_j)

    return e_ρ_plus_i, e_ϕ_plus_i, e_z_plus_i, e_ρ_minus_i, e_ϕ_minus_i, e_z_minus_i,
           e_ρ_plus_j, e_ϕ_plus_j, e_z_plus_j, e_ρ_minus_j, e_ϕ_minus_j, e_z_minus_j
end

function electric_radiation_mode_cylindrical_base_components_external_both_polarizations_and_reflected_two_atoms(ρ_i, ρ_j, ω, β, m::Integer, fiber)
    a = fiber.radius
    q = sqrt(ω^2 - β^2)

    V_1, V_2, M_1, M_2, L_1, L_2, η = electric_radiation_mode_cylindrical_base_components_external_coefficients(a, q, ω, β, m, fiber)
    V_1_reflected = -V_1
    V_2_reflected = -V_2

    h1mρ_i, h2mρ_i, dh1mρ_i, dh2mρ_i = electric_radiation_mode_cylindrical_base_components_external_hankel(m, q, ρ_i)
    h1mρ_j, h2mρ_j, dh1mρ_j, dh2mρ_j = electric_radiation_mode_cylindrical_base_components_external_hankel(m, q, ρ_j)
    
    h1mρ_i_reflected = (-1)^m * h1mρ_i
    h2mρ_i_reflected = (-1)^m * h2mρ_i
    dh1mρ_i_reflected = (-1)^m * dh1mρ_i
    dh2mρ_i_reflected = (-1)^m * dh2mρ_i
    h1mρ_j_reflected = (-1)^m * h1mρ_j
    h2mρ_j_reflected = (-1)^m * h2mρ_j
    dh1mρ_j_reflected = (-1)^m * dh1mρ_j
    dh2mρ_j_reflected = (-1)^m * dh2mρ_j

    C_1_plus, C_2_plus, D_1_plus, D_2_plus, A_plus = electric_radiation_mode_cylindrical_base_components_external_polarization_coefficients(1, a, q, ω, V_1, V_2, M_1, M_2, L_1, L_2, η)
    C_1_plus_r, C_2_plus_r, D_1_plus_r, D_2_plus_r, A_plus_r = electric_radiation_mode_cylindrical_base_components_external_polarization_coefficients(1, a, q, ω, V_1_reflected, V_2_reflected, M_1, M_2, L_1, L_2, η)
    e_ρ_plus_i, e_ϕ_plus_i, e_z_plus_i = _electric_radiation_mode_cylindrical_base_components_external(ρ_i, m, ω, β, q, C_1_plus, C_2_plus, D_1_plus, D_2_plus, A_plus, h1mρ_i, h2mρ_i, dh1mρ_i, dh2mρ_i)
    e_ρ_plus_j, e_ϕ_plus_j, e_z_plus_j = _electric_radiation_mode_cylindrical_base_components_external(ρ_j, m, ω, β, q, C_1_plus, C_2_plus, D_1_plus, D_2_plus, A_plus, h1mρ_j, h2mρ_j, dh1mρ_j, dh2mρ_j)
    e_ρ_plus_i_reflected, e_ϕ_plus_i_reflected, e_z_plus_i_reflected = _electric_radiation_mode_cylindrical_base_components_external(ρ_i, -m, ω, β, q, C_1_plus_r, C_2_plus_r, D_1_plus_r, D_2_plus_r, A_plus_r, h1mρ_i_reflected, h2mρ_i_reflected, dh1mρ_i_reflected, dh2mρ_i_reflected)
    e_ρ_plus_j_reflected, e_ϕ_plus_j_reflected, e_z_plus_j_reflected = _electric_radiation_mode_cylindrical_base_components_external(ρ_j, -m, ω, β, q, C_1_plus_r, C_2_plus_r, D_1_plus_r, D_2_plus_r, A_plus_r, h1mρ_j_reflected, h2mρ_j_reflected, dh1mρ_j_reflected, dh2mρ_j_reflected)

    C_1_minus, C_2_minus, D_1_minus, D_2_minus, A_minus = electric_radiation_mode_cylindrical_base_components_external_polarization_coefficients(-1, a, q, ω, V_1, V_2, M_1, M_2, L_1, L_2, η)
    C_1_minus_r, C_2_minus_r, D_1_minus_r, D_2_minus_r, A_minus_r = electric_radiation_mode_cylindrical_base_components_external_polarization_coefficients(-1, a, q, ω, V_1_reflected, V_2_reflected, M_1, M_2, L_1, L_2, η)
    e_ρ_minus_i, e_ϕ_minus_i, e_z_minus_i = _electric_radiation_mode_cylindrical_base_components_external(ρ_i, m, ω, β, q, C_1_minus, C_2_minus, D_1_minus, D_2_minus, A_minus, h1mρ_i, h2mρ_i, dh1mρ_i, dh2mρ_i)
    e_ρ_minus_j, e_ϕ_minus_j, e_z_minus_j = _electric_radiation_mode_cylindrical_base_components_external(ρ_j, m, ω, β, q, C_1_minus, C_2_minus, D_1_minus, D_2_minus, A_minus, h1mρ_j, h2mρ_j, dh1mρ_j, dh2mρ_j)
    e_ρ_minus_i_reflected, e_ϕ_minus_i_reflected, e_z_minus_i_reflected = _electric_radiation_mode_cylindrical_base_components_external(ρ_i, -m, ω, β, q, C_1_minus_r, C_2_minus_r, D_1_minus_r, D_2_minus_r, A_minus_r, h1mρ_i_reflected, h2mρ_i_reflected, dh1mρ_i_reflected, dh2mρ_i_reflected)
    e_ρ_minus_j_reflected, e_ϕ_minus_j_reflected, e_z_minus_j_reflected = _electric_radiation_mode_cylindrical_base_components_external(ρ_j, -m, ω, β, q, C_1_minus_r, C_2_minus_r, D_1_minus_r, D_2_minus_r, A_minus_r, h1mρ_j_reflected, h2mρ_j_reflected, dh1mρ_j_reflected, dh2mρ_j_reflected)

    return e_ρ_plus_i, e_ϕ_plus_i, e_z_plus_i, e_ρ_minus_i, e_ϕ_minus_i, e_z_minus_i,
           e_ρ_plus_j, e_ϕ_plus_j, e_z_plus_j, e_ρ_minus_j, e_ϕ_minus_j, e_z_minus_j,
           e_ρ_plus_i_reflected, e_ϕ_plus_i_reflected, e_z_plus_i_reflected,
           e_ρ_minus_i_reflected, e_ϕ_minus_i_reflected, e_z_minus_i_reflected,
           e_ρ_plus_j_reflected, e_ϕ_plus_j_reflected, e_z_plus_j_reflected,
           e_ρ_minus_j_reflected, e_ϕ_minus_j_reflected, e_z_minus_j_reflected
end

function electric_radiation_mode_cylindrical_base_components(ρ, ω, β, l, m, fiber)
    if ρ < fiber.radius
        return electric_radiation_mode_cylindrical_base_components_internal(ρ, ω, β, l, m, fiber)
    else
        return electric_radiation_mode_cylindrical_base_components_external(ρ, ω, β, l, m, fiber)
    end
end

function electric_radiation_field_cartesian_components(ρ, ϕ, ω, β, l::Integer, m::Integer, fiber::Fiber{T}) where {T<:Number}
    e_ρ, e_ϕ, e_z = electric_radiation_mode_cylindrical_base_components(ρ, ω, β, l, m, fiber)

    e_x = e_ρ * cos(ϕ) - e_ϕ * sin(ϕ)
    e_y = e_ρ * sin(ϕ) + e_ϕ * cos(ϕ)

    return e_x, e_y, e_z
end

function electric_radiation_mode(ρ, ϕ, ω, β, l::Integer, m::Integer, fiber::Fiber{T}) where {T<:Number}
    e_ρ, e_ϕ, e_z = electric_radiation_mode_cylindrical_base_components(ρ, ω, β, l, m, fiber)
    return [e_ρ * cos(ϕ) - e_ϕ * sin(ϕ), e_ρ * sin(ϕ) + e_ϕ * cos(ϕ), e_z]
end
