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
        error("Expected polarization axis to be 'x' or 'y', got: '$axis'.")
    end
end

LinearPolarization(axis::String) = LinearPolarization(only(axis))
LinearPolarization(sym::Symbol) = LinearPolarization(string(sym))

struct CircularPolarization <: Polarization
    l::Int
    function CircularPolarization(l::Int)
        l ∈ (-1, 1) || throw(DomainError(l, "Polarization index must be +1 or -1."))
        return new(l)
    end
end

polarization(polarization::LinearPolarization) = polarization.ϕ₀
polarization(polarization::CircularPolarization) = polarization.l

struct GuidedMode{P<:Polarization}
    fiber::Fiber
    polarization::Polarization
    f::Int
    function GuidedMode(fiber::Fiber, polarization::P, f::Int) where {P<:Polarization}
        f ∈ (-1, 1) || throw(DomainError(f, "Propagation direction must be +1 or -1."))
        return new{P}(fiber, polarization, f)
    end
end

function Base.show(io::IO, mode::GuidedMode)
    println(io, "Guided mode in")
    println(io, mode.fiber)
    println(io, mode.polarization)
    print(io, "Propagation direction index f = $(mode.f)")
end

direction(mode::GuidedMode) = mode.f
polarization(mode::GuidedMode) = polarization(mode.polarization)
frequency(mode::GuidedMode) = frequency(mode.fiber)
propagation_constant(mode::GuidedMode) = propagation_constant(mode.fiber)
propagation_constant_derivative(mode::GuidedMode) = propagation_constant_derivative(mode.fiber)
radius(mode::GuidedMode) = radius(mode.fiber)

abstract type ElectricField end

struct GuidedField{P<:Polarization} <: ElectricField
    mode::GuidedMode{P}
    power::Float64
end

function Base.show(io::IO, field::GuidedField)
    println(io, "Guided field in")
    println(io, field.mode.fiber)
    println(io, field.mode.polarization)
    println(io, "Propagation direction index f = $(field.mode.f)")
    print(io, "Power: $(field.power)")
end

direction(field::GuidedField) = direction(field.mode)
polarization(field::GuidedField) = polarization(field.mode)
frequency(field::GuidedField) = frequency(field.mode)
propagation_constant(field::GuidedField) = propagation_constant(field.mode)
propagation_constant_derivative(field::GuidedField) = propagation_constant_derivative(field.mode)
radius(field::GuidedField) = radius(field.mode)

"""
    electric_guided_mode_cylindrical_base_components(ρ::Real, fiber::Fiber)

Compute the underlying cylindrical components of the guided mode electric field used in the
expressions for both the quasilinear and quasicircular guided modes.

These components for ``\\rho < a`` are given by
```math
\\begin{aligned}
    e_{\\rho} &= A \\mathrm{i} \\frac{q}{p} \\frac{K_{1}(q a)}{J_{1}(p a)}
    [(1 - s) J_{0}(p \\rho) - (1 + s) J_{2}(p \\rho)] \\\\
    e_{\\phi} &= -A \\frac{q}{p} \\frac{K_{1}(q a)}{J_{1}(p a)}
    [(1 - s) J_{0}(p \\rho) + (1 + s) J_{2}(p \\rho)] \\\\
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
s = \\frac{\\frac{1}{p^2 a^2} + \\frac{1}{q^2 a^2}}{\\frac{J_{1}'(p a)}{p a J_{1}(p a)} 
+ \\frac{K_{1}'(q a)}{q a K_{1}(q a)}}.
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

function electric_guided_mode_profile_cylindrical_components(
    ρ::Real,
    ϕ::Real,
    mode::GuidedMode{LinearPolarization},
)
    f = direction(mode)
    ϕ₀ = polarization(mode)
    e_ρ, e_ϕ, e_z = electric_guided_mode_cylindrical_base_components(ρ, mode.fiber)
    e_ρ = sqrt(2) * e_ρ * cos(ϕ - ϕ₀)
    e_ϕ = sqrt(2) * im * e_ϕ * sin(ϕ - ϕ₀)
    e_z = sqrt(2) * f * e_z * cos(ϕ - ϕ₀)
    return e_ρ, e_ϕ, e_z
end

function electric_guided_mode_profile_cylindrical_components(
    ρ::Real,
    ϕ::Real,
    mode::GuidedMode{CircularPolarization},
)
    f = direction(mode)
    l = polarization(mode)
    e_ρ, e_ϕ, e_z = electric_guided_mode_cylindrical_base_components(ρ, mode.fiber)
    e_ρ = e_ρ * exp(im * l * ϕ)
    e_ϕ = l * e_ϕ * exp(im * l * ϕ)
    e_z = f * e_z * exp(im * l * ϕ)
    return e_ρ, e_ϕ, e_z
end

function electric_guided_mode_profile_cartesian_components(
    ρ::Real,
    ϕ::Real,
    mode::GuidedMode,
)
    e_ρ, e_ϕ, e_z = electric_guided_mode_profile_cylindrical_components(ρ, ϕ, mode)
    e_x = e_ρ * cos(ϕ) - e_ϕ * sin(ϕ)
    e_y = e_ρ * sin(ϕ) + e_ϕ * cos(ϕ)
    return e_x, e_y, e_z
end

function electric_guided_mode_profile_cartesian_components(
    ρ::Real,
    ϕ::Real,
    field::GuidedField,
)
    return electric_guided_mode_profile_cartesian_components(ρ, ϕ, field.mode)
end

"""
    electric_guided_field_cartesian_components(ρ::Real, ϕ::Real, z::Real, t::Real = 0,
                                               field::GuidedField)

Compute the cartesian components of a guided electric field at position ``(ρ, ϕ, z)`` and
time ``t`` with the given ``power``.
"""
function electric_guided_field_cartesian_components(
    ρ::Real,
    ϕ::Real,
    z::Real,
    t::Real,
    field::GuidedField,
)
    f = direction(field)
    ω = frequency(field)
    β = propagation_constant(field)
    dβ = propagation_constant_derivative(field)
    e_x, e_y, e_z = electric_guided_mode_profile_cartesian_components(ρ, ϕ, field)
    C = sqrt(field.power * dβ / 2)
    phase_factor = exp(im * (f * β * z - ω * t))
    e_x = C * e_x * phase_factor
    e_y = C * e_y * phase_factor
    e_z = C * e_z * phase_factor
    return e_x, e_y, e_z
end

function electric_guided_field_cartesian_components(
    ρ::Real,
    ϕ::Real,
    z::Real,
    field::GuidedField,
)
    return electric_guided_field_cartesian_components(ρ, ϕ, z, 0.0, field)
end

"""
    electric_guided_field_cartesian_vector(ρ::Real, ϕ::Real, z::Real, t::Real = 0,
                                           field::GuidedField)

Compute the guided electric field vector at position ``(ρ, ϕ, z)`` and time ``t`` with the
given ``power``.
"""
function electric_guided_field_cartesian_vector(
    ρ::Real,
    ϕ::Real,
    z::Real,
    t::Real,
    field::GuidedField,
)
    return collect(electric_guided_field_cartesian_components(ρ, ϕ, z, t, field))
end

function electric_guided_field_cartesian_vector(
    ρ::Real,
    ϕ::Real,
    z::Real,
    field::GuidedField,
)
    return collect(electric_guided_field_cartesian_components(ρ, ϕ, z, field))
end

struct ExternalField <: ElectricField
    wavelength::Float64
    frequency::Float64
    rabi_frequency::Float64
end

function Base.show(io::IO, field::ExternalField)
    println(io, "External mode with parameters:")
    println(io, "λ = $(field.wavelength)")
    println(io, "ω = $(field.frequency)")
    print(io, "Ω = $(field.rabi_frequency)")
end

struct RadiationAuxillaryCoefficients
    V::ComplexF64
    M::ComplexF64
    L::ComplexF64
    η::Float64
end

function Base.show(io::IO, coefficients::RadiationAuxillaryCoefficients)
    println(io, "Radiation auxillary coefficients:")
    println(io, "V = $(coefficients.V)")
    println(io, "M = $(coefficients.M)")
    println(io, "L = $(coefficients.L)")
    print(io,   "η  = $(coefficients.η)")
end

"""
    reflect(coefficients::RadiationAuxillaryCoefficients)

Compute the corresponding auxillary coefficents when sending the index m to -m.
"""
function reflect(coefficients::RadiationAuxillaryCoefficients)
    return RadiationAuxillaryCoefficients(-coefficients.V, 
                                          coefficients.M,
                                          coefficients.L,
                                          coefficients.η)
end

function radiation_mode_auxillary_coefficients(
    ω::Real,
    β::Real,
    m::Integer,
    q::Real,
    fiber::Fiber,
)
    a = fiber.radius
    n = fiber.refractive_index
    h = sqrt(n^2 * ω^2 - β^2)
    
    J = besselj(m, h * a)
    dJ = 0.5 * (besselj(m - 1, h * a) - besselj(m + 1, h * a))
    h₁ = hankelh1(m, q * a)
    dh₁ = 0.5 * (hankelh1(m - 1, q * a) - hankelh1(m + 1, q * a))

    V = m * ω * β / (a * h^2 * q^2) * (1 - n^2) * J * conj(h₁)
    M = 1 / h * dJ * conj(h₁) - 1 / q * J * conj(dh₁)
    L = n^2 / h * dJ * conj(h₁) - 1 / q * J * conj(dh₁)
    η = sqrt((abs2(V) + abs2(L)) / (abs2(V) + abs2(M)))

    return RadiationAuxillaryCoefficients(V, M, L, η)
end

struct RadiationBoundaryConditionCoefficients
    A::Float64
    B::ComplexF64
    C₁::ComplexF64
    C₂::ComplexF64
    D₁::ComplexF64
    D₂::ComplexF64
end

function Base.show(io::IO, coefficients::RadiationBoundaryConditionCoefficients)
    println(io, "Radiation boundary condition coefficients:")
    println(io, "A  = $(coefficients.A)")
    println(io, "B  = $(coefficients.B)")
    println(io, "C₁ = $(coefficients.C₁)")
    println(io, "C₂ = $(coefficients.C₂)")
    println(io, "D₁ = $(coefficients.D₁)")
    print(io,   "D₂ = $(coefficients.D₂)")
end

function radiation_mode_boundary_coefficients(
    a::Real,
    ω::Real,
    l::Integer,
    q::Real,
    coefficients::RadiationAuxillaryCoefficients,
)
    L = coefficients.L
    V = coefficients.V
    M = coefficients.M
    η = coefficients.η
    
    B = im * l * η
    factor = im * π * q^2 * a / 4
    C = -factor * (L + im * B * V)
    D = factor * (im * V - B * M)
    N_ν = 8π * ω / (q^2) * (abs2(C) + abs2(D))
    A = 1 / sqrt(N_ν)

    return RadiationBoundaryConditionCoefficients(A, B, C, conj(C), D, -conj(D))
end

struct RadiationHankelCoefficients
    h₁::ComplexF64
    h₂::ComplexF64
    dh₁::ComplexF64
    dh₂::ComplexF64
end

function Base.show(io::IO, coefficients::RadiationHankelCoefficients)
    println(io, "Radiation hankel coefficients:")
    println(io, "h₁  = $(coefficients.h₁)")
    println(io, "h₂  = $(coefficients.h₂)")
    println(io, "dh₁ = $(coefficients.dh₁)")
    print(io,   "dh₂ = $(coefficients.dh₂)")
end

function Base.:*(coefficients::RadiationHankelCoefficients, factor::Number)
    return RadiationHankelCoefficients(factor * coefficients.h₁,
                                       factor * coefficients.h₂,
                                       factor * coefficients.dh₁,
                                       factor * coefficients.dh₂)
end

function Base.:*(factor::Number, coefficients::RadiationHankelCoefficients)
    return RadiationHankelCoefficients(factor * coefficients.h₁,
                                       factor * coefficients.h₂,
                                       factor * coefficients.dh₁,
                                       factor * coefficients.dh₂)
end

"""
    reflect(coefficients::RadiationHankelCoefficients)

Compute the corresponding auxillary coefficents when sending the index m to -m.
"""
reflect(coefficients::RadiationHankelCoefficients, m::Integer) = (-1)^m * coefficients

function radiation_mode_hankel_coefficients(ρ::Float64, m::Integer, q::Float64)
    h₁ = hankelh1(m, q * ρ)
    dh₁ = 0.5 * (hankelh1(m - 1, q * ρ) - hankelh1(m + 1, q * ρ))

    return RadiationHankelCoefficients(h₁, conj(h₁), dh₁, conj(dh₁))
end

function electric_radiation_mode_cylindrical_base_components_internal(
    ρ::Float64,
    ω::Float64,
    β::Float64,
    m::Integer,
    l::Integer,
    fiber::Fiber,
)
    a = fiber.radius
    n = fiber.refractive_index
    h = sqrt(n^2 * ω^2 - β^2)
    q = sqrt(ω^2 - β^2)

    radiation_coefficients = radiation_mode_auxillary_coefficients(ω, β, m, q, fiber)
    boundary_coefficients = radiation_mode_boundary_coefficients(a, ω, l, q, radiation_coefficients)
    
    A = boundary_coefficients.A
    B = boundary_coefficients.B

    J = besselj(m, h * ρ)
    dJ = 1 / 2 * (besselj(m - 1, h * a) - besselj(m + 1, h * ρ))
    
    e_ρ = A * im / (h^2) * (β * h * dJ + im * m * ω / ρ * B * J)
    e_ϕ = A * im / (h^2) * (im * m * β / ρ * J - h * ω * B * dJ)
    e_z = A * J

    return e_ρ, e_ϕ, e_z
end

function electric_radiation_mode_cylindrical_base_components_external(
    ρ::Float64,
    ω::Float64,
    β::Float64,
    q::Float64,
    m::Integer, 
    boundary_coefficients::RadiationBoundaryConditionCoefficients,
    hankel_coefficients::RadiationHankelCoefficients,
)
    A = boundary_coefficients.A
    C₁ = boundary_coefficients.C₁
    C₂ = boundary_coefficients.C₂
    D₁ = boundary_coefficients.D₁
    D₂ = boundary_coefficients.D₂

    h₁ = hankel_coefficients.h₁
    h₂ = hankel_coefficients.h₂
    dh₁ = hankel_coefficients.dh₁
    dh₂ = hankel_coefficients.dh₂

    e_ρ = A * im / (q^2) * (  β * q * C₁ * dh₁ + im * m * ω / ρ * D₁ * h₁
                            + β * q * C₂ * dh₂ + im * m * ω / ρ * D₂ * h₂)
    e_ϕ = A * im / (q^2) * (im * m * β / ρ * C₁ * h₁ - q * ω * D₁ * dh₁
                            + im * m * β / ρ * C₂ * h₂ - q * ω * D₂ * dh₂)
    e_z = A * (C₁ * h₁ + C₂ * h₂)

    return e_ρ, e_ϕ, e_z
end

function electric_radiation_mode_cylindrical_base_components_external(
    ρ::Float64,
    ω::Float64,
    β::Float64,
    m::Integer, 
    l::Integer,
    fiber::Fiber,
)
    a = fiber.radius
    q = sqrt(ω^2 - β^2)

    radiation_coefficients = radiation_mode_auxillary_coefficients(ω, β, m, q, fiber)
    boundary_coefficients = radiation_mode_boundary_coefficients(a, ω, l, q, radiation_coefficients)
    hankel_coefficients = radiation_mode_hankel_coefficients(ρ, m, q)

    return electric_radiation_mode_cylindrical_base_components_external(ρ, ω, β, q, m, boundary_coefficients, hankel_coefficients)
end

function electric_radiation_mode_cylindrical_base_components(
    ρ::Float64,
    ω::Float64,
    β::Float64,
    m::Integer,
    l::Integer,
    fiber::Fiber,
)
    if ρ < fiber.radius
        return electric_radiation_mode_cylindrical_base_components_internal(ρ, ω, β, m, l, fiber)
    else
        return electric_radiation_mode_cylindrical_base_components_external(ρ, ω, β, m, l, fiber)
    end
end

function electric_radiation_field_cartesian_components(
    ρ::Float64,
    ϕ::Float64,
    ω::Float64,
    β::Float64,
    m::Integer,
    l::Integer,
    fiber::Fiber,
)
    e_ρ, e_ϕ, e_z = electric_radiation_mode_cylindrical_base_components(ρ, ω, β, m, l, fiber)

    e_x = e_ρ * cos(ϕ) - e_ϕ * sin(ϕ)
    e_y = e_ρ * sin(ϕ) + e_ϕ * cos(ϕ)

    return e_x, e_y, e_z
end

function electric_radiation_mode_cylindrical_base_components_external_both_polarizations_two_atoms(
    ρ_i::Float64,
    ρ_j::Float64,
    ω::Float64,
    β::Float64,
    m::Integer,
    fiber::Fiber,
)
    a = fiber.radius
    q = sqrt(ω^2 - β^2)

    aux_coeffs = radiation_mode_auxillary_coefficients(ω, β, m, q, fiber)

    h_i = radiation_mode_hankel_coefficients(ρ_i, m, q)
    h_j = radiation_mode_hankel_coefficients(ρ_j, m, q)
    
    boundary_coeffs_plus = radiation_mode_boundary_coefficients(a, ω, 1, q, aux_coeffs)
    e_ρ_plus_i, e_ϕ_plus_i, e_z_plus_i = electric_radiation_mode_cylindrical_base_components_external(ρ_i, ω, β, q, m, boundary_coeffs_plus, h_i)
    e_ρ_plus_j, e_ϕ_plus_j, e_z_plus_j = electric_radiation_mode_cylindrical_base_components_external(ρ_j, ω, β, q, m, boundary_coeffs_plus, h_j)
    
    boundary_coeffs_minus = radiation_mode_boundary_coefficients(a, ω, -1, q, aux_coeffs)
    e_ρ_minus_i, e_ϕ_minus_i, e_z_minus_i = electric_radiation_mode_cylindrical_base_components_external(ρ_i, ω, β, q, m, boundary_coeffs_minus, h_i)
    e_ρ_minus_j, e_ϕ_minus_j, e_z_minus_j = electric_radiation_mode_cylindrical_base_components_external(ρ_i, ω, β, q, m, boundary_coeffs_minus, h_i)

    return e_ρ_plus_i, e_ϕ_plus_i, e_z_plus_i, e_ρ_minus_i, e_ϕ_minus_i, e_z_minus_i,
           e_ρ_plus_j, e_ϕ_plus_j, e_z_plus_j, e_ρ_minus_j, e_ϕ_minus_j, e_z_minus_j
end

function electric_radiation_mode_cylindrical_base_components_external_both_polarizations_and_reflected_two_atoms(
    ρ_i::Float64,
    ρ_j::Float64,
    ω::Float64,
    β::Float64,
    m::Integer,
    fiber::Fiber,
)
    a = fiber.radius
    q = sqrt(ω^2 - β^2)

    aux_coeffs = radiation_mode_auxillary_coefficients(ω, β, m, q, fiber)
    aux_coeffs_reflected = reflect(aux_coeffs)

    h_i = radiation_mode_hankel_coefficients(ρ_i, m, q)
    h_j = radiation_mode_hankel_coefficients(ρ_j, m, q)
    h_i_reflected = reflect(h_i, m)
    h_j_reflected = reflect(h_j, m)

    boundary_coeffs_plus = radiation_mode_boundary_coefficients(a, ω, 1, q, aux_coeffs)
    boundary_coeffs_plus_reflected = radiation_mode_boundary_coefficients(a, ω, 1, q, aux_coeffs_reflected)

    e_ρ_plus_i, e_ϕ_plus_i, e_z_plus_i = electric_radiation_mode_cylindrical_base_components_external(ρ_i, ω, β, q, m, boundary_coeffs_plus, h_i)
    e_ρ_plus_j, e_ϕ_plus_j, e_z_plus_j = electric_radiation_mode_cylindrical_base_components_external(ρ_j, ω, β, q, m, boundary_coeffs_plus, h_j)
    e_ρ_plus_i_reflected, e_ϕ_plus_i_reflected, e_z_plus_i_reflected = electric_radiation_mode_cylindrical_base_components_external(ρ_i, ω, β, q, -m, boundary_coeffs_plus_reflected, h_i_reflected)
    e_ρ_plus_j_reflected, e_ϕ_plus_j_reflected, e_z_plus_j_reflected = electric_radiation_mode_cylindrical_base_components_external(ρ_j, ω, β, q, -m, boundary_coeffs_plus_reflected, h_j_reflected)
    
    boundary_coeffs_minus = radiation_mode_boundary_coefficients(a, ω, -1, q, aux_coeffs)
    boundary_coeffs_minus_reflected = radiation_mode_boundary_coefficients(a, ω, -1, q, aux_coeffs_reflected)

    e_ρ_minus_i, e_ϕ_minus_i, e_z_minus_i = electric_radiation_mode_cylindrical_base_components_external(ρ_i, ω, β, q, m, boundary_coeffs_minus, h_i)
    e_ρ_minus_j, e_ϕ_minus_j, e_z_minus_j = electric_radiation_mode_cylindrical_base_components_external(ρ_j, ω, β, q, m, boundary_coeffs_minus, h_j)
    e_ρ_minus_i_reflected, e_ϕ_minus_i_reflected, e_z_minus_i_reflected = electric_radiation_mode_cylindrical_base_components_external(ρ_i, ω, β, q, -m, boundary_coeffs_minus_reflected, h_i_reflected)
    e_ρ_minus_j_reflected, e_ϕ_minus_j_reflected, e_z_minus_j_reflected = electric_radiation_mode_cylindrical_base_components_external(ρ_j, ω, β, q, -m, boundary_coeffs_minus_reflected, h_j_reflected)
    
    return e_ρ_plus_i, e_ϕ_plus_i, e_z_plus_i, e_ρ_minus_i, e_ϕ_minus_i, e_z_minus_i,
           e_ρ_plus_j, e_ϕ_plus_j, e_z_plus_j, e_ρ_minus_j, e_ϕ_minus_j, e_z_minus_j,
           e_ρ_plus_i_reflected, e_ϕ_plus_i_reflected, e_z_plus_i_reflected,
           e_ρ_minus_i_reflected, e_ϕ_minus_i_reflected, e_z_minus_i_reflected,
           e_ρ_plus_j_reflected, e_ϕ_plus_j_reflected, e_z_plus_j_reflected,
           e_ρ_minus_j_reflected, e_ϕ_minus_j_reflected, e_z_minus_j_reflected
end

# Here be dragons
struct RadiationSurfaceCoefficients
    J::ComplexF64
    dJ::ComplexF64
    h₁::ComplexF64
    dh₁::ComplexF64
end

function Base.show(io::IO, coefficients::RadiationSurfaceCoefficients)
    println(io, "Radiation hankel coefficients:")
    println(io, "J   = $(coefficients.J)")
    println(io, "dJ  = $(coefficients.dJ)")
    println(io, "h₁  = $(coefficients.h₁)")
    print(io,   "dh₁ = $(coefficients.dh₁)")
end

function radiation_mode_surface_coefficients(a, m, h, q)
    J = besselj(m, h * a)
    dJ = 0.5 * (besselj(m - 1, h * a) - besselj(m + 1, h * a))
    h₁ = hankelh1(m, q * a)
    dh₁ = 0.5 * (hankelh1(m - 1, q * a) - hankelh1(m + 1, q * a))

    return RadiationSurfaceCoefficients(J, dJ, h₁, dh₁)
end

function radiation_mode_auxillary_coefficients_efficient(ω::Real, β::Real, m::Integer, 
                                                         h::Real, q::Real, fiber::Fiber,
                                                         surface_coefficients)
    a = fiber.radius
    n = fiber.refractive_index

    J = surface_coefficients.J
    dJ = surface_coefficients.dJ
    h₁ = surface_coefficients.h₁
    dh₁ = surface_coefficients.dh₁

    V = m * ω * β / (a * h^2 * q^2) * (1 - n^2) * J * conj(h₁)
    M = 1 / h * dJ * conj(h₁) - 1 / q * J * conj(dh₁)
    L = n^2 / h * dJ * conj(h₁) - 1 / q * J * conj(dh₁)
    η = sqrt((abs2(V) + abs2(L)) / (abs2(V) + abs2(M)))

    return RadiationAuxillaryCoefficients(V, M, L, η)
end

function electric_radiation_mode_cylindrical_base_components_external_both_polarizations_and_reflected(ρ, ω, β, q, m::Integer, fiber)
    a = fiber.radius

    aux_coeffs = radiation_mode_auxillary_coefficients(ω, β, m, q, fiber)
    aux_coeffs_r = reflect(aux_coeffs)

    h = radiation_mode_hankel_coefficients(ρ, m, q)
    h_r = reflect(h, m)

    boundary_coeffs_plus = radiation_mode_boundary_coefficients(a, ω, 1, q, aux_coeffs)
    boundary_coeffs_plus_r = radiation_mode_boundary_coefficients(a, ω, 1, q, aux_coeffs_r)

    e_ρ_plus, e_ϕ_plus, e_z_plus = electric_radiation_mode_cylindrical_base_components_external(ρ, ω, β, q, m, boundary_coeffs_plus, h)
    e_ρ_plus_r, e_ϕ_plus_r, e_z_plus_r = electric_radiation_mode_cylindrical_base_components_external(ρ, ω, β, q, -m, boundary_coeffs_plus_r, h_r)
    
    boundary_coeffs_minus = radiation_mode_boundary_coefficients(a, ω, -1, q, aux_coeffs)
    boundary_coeffs_minus_r = radiation_mode_boundary_coefficients(a, ω, -1, q, aux_coeffs_r)

    e_ρ_minus, e_ϕ_minus, e_z_minus = electric_radiation_mode_cylindrical_base_components_external(ρ, ω, β, q, m, boundary_coeffs_minus, h)
    e_ρ_minus_r, e_ϕ_minus_r, e_z_minus_r = electric_radiation_mode_cylindrical_base_components_external(ρ, ω, β, q, -m, boundary_coeffs_minus_r, h_r)
    
    return e_ρ_plus, e_ϕ_plus, e_z_plus, e_ρ_minus, e_ϕ_minus, e_z_minus,
           e_ρ_plus_r, e_ϕ_plus_r, e_z_plus_r, e_ρ_minus_r, e_ϕ_minus_r, e_z_minus_r
end

function surface_evaluations(m_max, hs, qs, a, resolution)
    Jhas = Matrix{Float64}(undef, m_max + 2, resolution)
    Jqas = Matrix{Float64}(undef, m_max + 2, resolution)
    Yqas = Matrix{Float64}(undef, m_max + 2, resolution)

    for m in 0:m_max + 1
        for k in 1:resolution
            Jhas[m + 1, k] = besselj(m, hs[k] * a)
            Jqas[m + 1, k] = besselj(m, qs[k] * a)
            Yqas[m + 1, k] = bessely(m, qs[k] * a)
        end
    end

    dJhas = Matrix{Float64}(undef, m_max + 1, resolution)
    dJqas = Matrix{Float64}(undef, m_max + 1, resolution)
    dYqas = Matrix{Float64}(undef, m_max + 1, resolution)

    dJhas[1, :] = -Jhas[2, :]
    dJqas[1, :] = -Jqas[2, :]
    dYqas[1, :] = -Yqas[2, :]

    for m in 1:m_max
        dJhas[m + 1, :] = 0.5 * (Jhas[m, :] - Jhas[m + 2, :])
        dJqas[m + 1, :] = 0.5 * (Jqas[m, :] - Jqas[m + 2, :])
        dYqas[m + 1, :] = 0.5 * (Yqas[m, :] - Yqas[m + 2, :])
    end

    h₁qas = Jqas + im * Yqas
    dh₁qas = dJqas + im * dYqas

    return Jhas, dJhas, h₁qas, dh₁qas
end
