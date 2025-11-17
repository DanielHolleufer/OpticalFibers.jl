## Guided Modes ##

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

struct GuidedMode{P<:Polarization,T<:Real}
    fiber::Fiber{T}
    polarization::Polarization
    f::Int
    function GuidedMode(fiber::Fiber{T}, polarization::P, f::Int) where {P<:Polarization,T<:Real}
        f ∈ (-1, 1) || throw(DomainError(f, "Propagation direction must be +1 or -1."))
        return new{P,T}(fiber, polarization, f)
    end
end

function Base.show(io::IO, mode::GuidedMode)
    println(io, "Guided mode in")
    println(io, mode.fiber)
    println(io, mode.polarization)
    print(io, "Propagation direction index f = $(mode.f)")
end

wavelength(mode::GuidedMode) = wavelength(mode.fiber)
direction(mode::GuidedMode) = mode.f
polarization(mode::GuidedMode) = polarization(mode.polarization)
frequency(mode::GuidedMode) = frequency(mode.fiber)
propagation_constant(mode::GuidedMode) = propagation_constant(mode.fiber)
function propagation_constant_derivative(mode::GuidedMode)
    return propagation_constant_derivative(mode.fiber)
end
radius(mode::GuidedMode) = radius(mode.fiber)

abstract type ElectricField end

struct GuidedField{P<:Polarization,T<:Real} <: ElectricField
    mode::GuidedMode{P,T}
    power::T
end

function Base.show(io::IO, field::GuidedField)
    println(io, "Guided field in")
    println(io, field.mode.fiber)
    println(io, field.mode.polarization)
    println(io, "Propagation direction index f = $(field.mode.f)")
    print(io, "Power: $(field.power)")
end

wavelength(field::GuidedField) = wavelength(field.mode)
direction(field::GuidedField) = direction(field.mode)
polarization(field::GuidedField) = polarization(field.mode)
frequency(field::GuidedField) = frequency(field.mode)
propagation_constant(field::GuidedField) = propagation_constant(field.mode)
function propagation_constant_derivative(field::GuidedField) 
    return propagation_constant_derivative(field.mode)
end
radius(field::GuidedField) = radius(field.mode)

"""
    electric_guided_mode_base(ρ::Real, fiber::Fiber)

Compute the underlying cylindrical components of the guided mode
electric field used in the expressions for both the quasilinear and
quasicircular guided modes.

These components for ``\\rho < a`` are given by
```math
\\begin{aligned}
    e_{\\rho} &= A \\mathrm{i} \\frac{q}{h} \\frac{K_{1}(q a)}{J_{1}(h a)}
    [(1 - s) J_{0}(h \\rho) - (1 + s) J_{2}(h \\rho)] \\\\
    e_{\\phi} &= -A \\frac{q}{h} \\frac{K_{1}(q a)}{J_{1}(h a)}
    [(1 - s) J_{0}(h \\rho) + (1 + s) J_{2}(h \\rho)] \\\\
    e_{z} &= A \\frac{2 q}{\\beta} \\frac{K_{1}(q a)}{J_{1}(h a)} J_{1}(h \\rho),
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
where ``A`` is the normalization constant, ``a`` is the fiber radius,
``\\beta`` is the propagation constant, ``h = \\sqrt{n^2 k^2 - \\beta^2}``,
and ``q = \\sqrt{\\beta^2 - k^2}``, with ``k`` being the free space
wavenumber of the light. Futhermore, ``J_n`` and ``K_n`` are Bessel
functions of the first kind, and modified Bessel functions of the
second kind, respectively, and the prime denotes the derivative.
Lastly, ``s`` is defined as
```math
s = \\frac{\\frac{1}{h^2 a^2} + \\frac{1}{q^2 a^2}}{\\frac{J_{1}'(h a)}{h a J_{1}(h a)} 
+ \\frac{K_{1}'(q a)}{q a K_{1}(q a)}}.
```
"""
function electric_guided_mode_base(ρ::Real, fiber::Fiber)
    ρ ≥ 0.0 || throw(DomainError(ρ, "Radial coordinate must be non-negative."))

    a = radius(fiber)
    β = propagation_constant(fiber)
    h = fiber.internal_parameter
    q = fiber.external_parameter
    s = fiber.s_parameter
    A = fiber.normalization_constant

    if ρ < a
        K1J1 = besselk1(q * a) / besselj1(h * a)
        j0 = besselj0(h * ρ)
        j1 = besselj1(h * ρ)
        j2 = besselj(2, h * ρ)
        e_ρ = im * A * q / h * K1J1 * ((1 - s) * j0 - (1 + s) * j2)
        e_ϕ = -A * q / h * K1J1 * ((1 - s) * j0 + (1 + s) * j2)
        e_z = 2 * A * q / β * K1J1 * j1
    else
        k0 = besselk0(q * ρ)
        k1 = besselk1(q * ρ)
        k2 = besselk(2, q * ρ)
        e_ρ = im * A * ((1 - s) * k0 + (1 + s) * k2)
        e_ϕ = -A * ((1 - s) * k0 - (1 + s) * k2)
        e_z = 2 * A * q / β * k1
    end

    return e_ρ, e_ϕ, e_z
end

function electric_guided_mode_profile_cylindrical(
    ρ::Real, ϕ::Real, mode::GuidedMode{LinearPolarization}
)
    f = direction(mode)
    ϕ₀ = polarization(mode)
    e_ρ, e_ϕ, e_z = electric_guided_mode_base(ρ, mode.fiber)
    e_ρ = sqrt(2) * e_ρ * cos(ϕ - ϕ₀)
    e_ϕ = sqrt(2) * im * e_ϕ * sin(ϕ - ϕ₀)
    e_z = sqrt(2) * f * e_z * cos(ϕ - ϕ₀)

    return e_ρ, e_ϕ, e_z
end

function electric_guided_mode_profile_cylindrical(
    ρ::Real, ϕ::Real, mode::GuidedMode{CircularPolarization}
)
    f = direction(mode)
    l = polarization(mode)
    e_ρ, e_ϕ, e_z = electric_guided_mode_base(ρ, mode.fiber)
    e_ρ = e_ρ * exp(im * l * ϕ)
    e_ϕ = l * e_ϕ * exp(im * l * ϕ)
    e_z = f * e_z * exp(im * l * ϕ)

    return e_ρ, e_ϕ, e_z
end

function electric_guided_mode_profile_cartesian(ρ::Real, ϕ::Real, mode::GuidedMode)
    e_ρ, e_ϕ, e_z = electric_guided_mode_profile_cylindrical(ρ, ϕ, mode)
    e_x = e_ρ * cos(ϕ) - e_ϕ * sin(ϕ)
    e_y = e_ρ * sin(ϕ) + e_ϕ * cos(ϕ)

    return e_x, e_y, e_z
end

function electric_guided_mode_profile_cartesian(ρ::Real, ϕ::Real, field::GuidedField)
    return electric_guided_mode_profile_cartesian(ρ, ϕ, field.mode)
end

"""
    electric_guided_field_cartesian(ρ::Real, ϕ::Real, z::Real, t::Real, field::GuidedField)

Compute the cartesian components of a guided electric field at
position ``(ρ, ϕ, z)`` and time ``t`` with the given ``power``.
"""
function electric_guided_field_cartesian(
    ρ::Real, ϕ::Real, z::Real, t::Real, field::GuidedField
)
    f = direction(field)
    ω = frequency(field)
    β = propagation_constant(field)
    dβ = propagation_constant_derivative(field)

    e_x, e_y, e_z = electric_guided_mode_profile_cartesian(ρ, ϕ, field)
    C = sqrt(field.power * dβ / 2)
    phase_factor = exp(im * (f * β * z - ω * t))
    e_x = C * e_x * phase_factor
    e_y = C * e_y * phase_factor
    e_z = C * e_z * phase_factor

    return e_x, e_y, e_z
end

function electric_guided_field_cartesian(ρ::Real, ϕ::Real, z::Real, field::GuidedField)
    return electric_guided_field_cartesian(ρ, ϕ, z, 0.0, field)
end


## External Fields ##

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


## Radiation Modes ##

function bessel_hankel_surface_evaluations(m_max, hs, qs, a, resolution)
    J = Matrix{Float64}(undef, m_max + 1, resolution)
    dJ = Matrix{Float64}(undef, m_max + 1, resolution)
    H1 = Matrix{ComplexF64}(undef, m_max + 1, resolution)
    dH1 = Matrix{ComplexF64}(undef, m_max + 1, resolution)

    for j in 1:resolution
        J[1, j] = besselj0.(hs[j] * a)
        dJ[1, j] = -besselj1.(hs[j] * a)
        H1[1, j] = hankelh1.(0, qs[j] * a)
        dH1[1, j] = -hankelh1.(1, qs[j] * a)

        for i in 2:m_max + 1
            m = i - 1
            J[i, j] = besselj.(m, hs[j] * a)
            H1[i, j] = hankelh1.(m, qs[j] * a)
        end

        for i in 2:m_max
            dJ[i, j] = 0.5 * (J[i - 1, j] - J[i + 1, j])
            dH1[i, j] = 0.5 * (H1[i - 1, j] - H1[i + 1, j])
        end
        dJ[m_max + 1, j] = 0.5 * (J[m_max, j] - besselj.(m_max + 1, hs[j] * a))
        dH1[m_max + 1, j] = 0.5 * (H1[m_max, j] - hankelh1.(m_max + 1, qs[j] * a))
    end

    return (J = J, dJ = dJ, H1 = H1, dH1 = dH1)
end

struct RadiationAuxiliaryCoefficients
    V::ComplexF64
    M::ComplexF64
    L::ComplexF64
    η::Float64
end

function Base.show(io::IO, coefficients::RadiationAuxiliaryCoefficients)
    println(io, "Radiation auxiliary coefficients:")
    println(io, "V = $(coefficients.V)")
    println(io, "M = $(coefficients.M)")
    println(io, "L = $(coefficients.L)")
    print(io,   "η  = $(coefficients.η)")
end

function radiation_auxiliary_coefficients(
    ω::Real,
    β::Real,
    m::Integer,
    h::Real,
    q::Real,
    a::Real,
    n::Real,
    J::Real,
    dJ::Real,
    H1::Complex,
    dH1::Complex,
)
    H2 = conj(H1)
    dH2 = conj(dH1)
    
    V = m * ω * β / (a * h^2 * q^2) * (1 - n^2) * J * H2
    M = 1 / h * dJ * H2 - 1 / q * J * dH2
    L = n^2 / h * dJ * H2 - 1 / q * J * dH2
    η = sqrt((abs2(V) + abs2(L)) / (abs2(V) + abs2(M)))

    if isnan(η)
        η = 1.0
    end

    return RadiationAuxiliaryCoefficients(V, M, L, η)
end

function radiation_auxiliary_coefficients(
    ω::Real,
    βs::AbstractArray{<:Real},
    m_max::Integer,
    hs::AbstractArray{<:Real},
    qs::AbstractArray{<:Real},
    a::Real,
    n::Real,
    bessel_hankel_evaluations::NamedTuple{(:J, :dJ, :H1, :dH1)},
)
    Nm = length(0:m_max)
    Nβ = length(βs)
    auxiliary = Matrix{RadiationAuxiliaryCoefficients}(undef, Nm, Nβ)
    for (i, m) in enumerate(0:m_max)
        for (j, β) in enumerate(βs)
            h = hs[j]
            q = qs[j]
            J = bessel_hankel_evaluations.J[i, j]
            dJ = bessel_hankel_evaluations.dJ[i, j]
            H1 = bessel_hankel_evaluations.H1[i, j]
            dH1 = bessel_hankel_evaluations.dH1[i, j]
            auxiliary[i, j] = radiation_auxiliary_coefficients(
                ω, β, m, h, q, a, n, J, dJ, H1, dH1
            )
        end
    end

    return auxiliary
end

struct RadiationBoundaryCoefficients
    A::Float64
    B::ComplexF64
    C::ComplexF64
    D::ComplexF64
end

function Base.show(io::IO, coefficients::RadiationBoundaryCoefficients)
    println(io, "Radiation boundary condition coefficients:")
    println(io, "A = $(coefficients.A)")
    println(io, "B = $(coefficients.B)")
    println(io, "C = $(coefficients.C)")
    print(io,   "D = $(coefficients.D)")
end

function radiation_boundary_coefficients(
    a::Real, ω::Real, l::Integer, q::Real, coefficients::RadiationAuxiliaryCoefficients
)
    V = coefficients.V
    M = coefficients.M
    L = coefficients.L
    η = coefficients.η
    
    B = im * l * η
    factor = im * π * q^2 * a / 4
    C = -factor * (L + im * B * V)
    D = factor * (im * V - B * M)
    N_ν = 8π * ω / (q^2) * (abs2(C) + abs2(D))
    A = 1 / sqrt(N_ν)

    return RadiationBoundaryCoefficients(A, B, C, D)
end

function radiation_boundary_coefficients(
    a::Real,
    ω::Real,
    β::Real,
    m::Integer,
    l::Integer,
    q::Real,
    coefficients::RadiationAuxiliaryCoefficients,
)
    V = sign(m) * sign(β) * coefficients.V
    M = coefficients.M
    L = coefficients.L
    η = coefficients.η
    
    B = im * l * η
    factor = im * π * q^2 * a / 4
    C = -factor * (L + im * B * V)
    D = factor * (im * V - B * M)
    N_ν = 8π * ω / (q^2) * (abs2(C) + abs2(D))
    A = 1 / sqrt(N_ν)

    return RadiationBoundaryCoefficients(A, B, C, D)
end

function radiation_boundary_coefficients(
    ω::Real,
    m_max::Integer,
    βs::AbstractArray{<:Real},
    qs::AbstractArray{<:Real},
    a::Real,
    auxiliary_coefficients::Matrix{RadiationAuxiliaryCoefficients},
)
    ls = (-1, 1)
    ms = -m_max:m_max
    Nm = length(ms)
    Nβ = Int(length(βs) / 2)
    Nl = length(ls)
    boundary = Array{RadiationBoundaryCoefficients,3}(undef, Nm, 2 * Nβ, Nl)
    for (k, l) in enumerate(ls)
        for (j, β) in enumerate(βs)
            j_auxiliary = maximum((Nβ + 1 - j, j - Nβ))
            for (i, m) in enumerate(ms)
                i_auxiliary = abs(m) + 1
                q = qs[j_auxiliary]
                aux = auxiliary_coefficients[i_auxiliary, j_auxiliary]
                boundary[i, j, k] = radiation_boundary_coefficients(a, ω, β, m, l, q, aux)
            end
        end
    end

    return boundary
end

function radiation_boundary_coefficients(
    m_max::Integer,
    βs::AbstractArray{<:Real},
    hs::AbstractArray{<:Real},
    qs::AbstractArray{<:Real},
    fiber::Fiber,
)
    a = radius(fiber)
    ω = frequency(fiber)
    n = refractive_index(fiber)
    evaluations = bessel_hankel_surface_evaluations(m_max, hs, qs, a, length(hs))
    auxiliary = radiation_auxiliary_coefficients(
        ω, view(βs, length(hs) + 1:length(βs)), m_max, hs, qs, a, n, evaluations
    )
    return radiation_boundary_coefficients(ω, m_max, βs, qs, a, auxiliary)
end

function hankel_evaluations!(
    H1::Matrix{ComplexF64},
    dH1::Matrix{ComplexF64},
    ρ::Real,
    m_max::Integer,
    qs::AbstractVector{<:Real}
)
    for j in eachindex(qs)
        qρ = qs[j] * ρ
        for i in 1:m_max + 1
            H1[i, j] = hankelh1(i - 1, qρ)
        end

        dH1[1, j] = -H1[2, j]
        for i in 2:m_max
            dH1[i, j] = 0.5 * (H1[i - 1, j] - H1[i + 1, j])
        end
        dH1[m_max + 1, j] = 0.5 * (H1[m_max, j] - hankelh1(m_max + 1, qρ))
    end
end

function hankel_evaluations(ρ::Real, m_max::Integer, qs::AbstractVector{<:Real})
    H1 = Matrix{ComplexF64}(undef, m_max + 1, length(qs))
    dH1 = similar(H1)
    hankel_evaluations!(H1, dH1, ρ, m_max, qs)
    return H1, dH1
end

function electric_radiation_mode_base_internal(
    ρ::Float64,
    ω::Float64,
    β::Float64,
    h::Float64,
    m::Integer,
    boundary_coefficients::RadiationBoundaryCoefficients,
)
    A = boundary_coefficients.A
    B = boundary_coefficients.B

    J = besselj(m, h * ρ)
    dJ = 1 / 2 * (besselj(m - 1, h * ρ) - besselj(m + 1, h * ρ))
    
    e_ρ = A * im / (h^2) * (β * h * dJ + im * m * ω / ρ * B * J)
    e_ϕ = A * im / (h^2) * (im * m * β / ρ * J - h * ω * B * dJ)
    e_z = A * J

    return e_ρ, e_ϕ, e_z
end

function electric_radiation_mode_base_internal(
    ρ::Float64, ω::Float64, β::Float64, m::Integer, l::Integer, fiber::Fiber
)
    a = fiber.radius
    n = fiber.refractive_index
    h = sqrt(n^2 * ω^2 - β^2)
    q = sqrt(ω^2 - β^2)

    J = besselj(m, h * a)
    dJ = 0.5 * (besselj(m - 1, h * a) - besselj(m + 1, h * a))
    H1 = hankelh1(m, q * a)
    dH1 = 0.5 * (hankelh1(m - 1, q * a) - hankelh1(m + 1, q * a))
    
    auxiliary = radiation_auxiliary_coefficients(ω, β, m, h, q, a, n, J, dJ, H1, dH1)
    boundary = radiation_boundary_coefficients(a, ω, l, q, auxiliary)

    return electric_radiation_mode_base_internal(ρ, ω, β, h, m, boundary)
end

function electric_radiation_mode_base_external(
    ρ::Float64,
    ω::Float64,
    β::Float64,
    q::Float64,
    m::Integer,
    H1::ComplexF64,
    dH1::ComplexF64,
    boundary_coefficients::RadiationBoundaryCoefficients,
)
    A = boundary_coefficients.A
    C = boundary_coefficients.C
    D = boundary_coefficients.D
    
    e_ρ = 2A * im / (q^2) * (β * q * real(C * dH1) - m * ω / ρ * imag(D * H1))
    e_ϕ = -2A / (q^2) * (m * β / ρ * real(C * H1) - q * ω * imag(D * dH1))
    e_z = 2A * real(C * H1)

    return e_ρ, e_ϕ, e_z
end

function electric_radiation_mode_base_external(
    ρ::Float64, ω::Float64, β::Float64, m::Integer, l::Integer, fiber::Fiber
)
    a = fiber.radius
    n = fiber.refractive_index
    h = sqrt(n^2 * ω^2 - β^2)
    q = sqrt(ω^2 - β^2)

    J = besselj(m, h * a)
    dJ = 0.5 * (besselj(m - 1, h * a) - besselj(m + 1, h * a))
    H1a = hankelh1(m, q * a)
    dH1a = 0.5 * (hankelh1(m - 1, q * a) - hankelh1(m + 1, q * a))

    radiation = radiation_auxiliary_coefficients(ω, β, m, h, q, a, n, J, dJ, H1a, dH1a)
    boundary = radiation_boundary_coefficients(a, ω, l, q, radiation)

    H1 = hankelh1(m, q * ρ)
    dH1 = 0.5 * (hankelh1(m - 1, q * ρ) - hankelh1(m + 1, q * ρ))

    return electric_radiation_mode_base_external(ρ, ω, β, q, m, H1, dH1, boundary)
end

function electric_radiation_mode_base(
    ρ::Float64, ω::Float64, β::Float64, m::Integer, l::Integer, fiber::Fiber
)
    if ρ < fiber.radius
        return electric_radiation_mode_base_internal(ρ, ω, β, m, l, fiber)
    else
        return electric_radiation_mode_base_external(ρ, ω, β, m, l, fiber)
    end
end

function electric_radiation_mode_base(
    ρ::Float64,
    ω::Float64,
    β::Float64,
    h::Float64,
    q::Float64,
    m::Integer,
    fiber::Fiber,
    boundary_coefficients::RadiationBoundaryCoefficients,
)
    if ρ < fiber.radius
        return electric_radiation_mode_base_internal(ρ, ω, β, h, m, boundary_coefficients)
    else
        return electric_radiation_mode_base_external(ρ, ω, β, q, m, boundary_coefficients)
    end
end

function electric_radiation_mode_profile_cartesian(
    ρ::Float64, ϕ::Float64, ω::Float64, β::Float64, m::Integer, l::Integer, fiber::Fiber
)
    e_ρ, e_ϕ, e_z = electric_radiation_mode_base(ρ, ω, β, m, l, fiber) .* exp(im * m * ϕ)

    e_x = e_ρ * cos(ϕ) - e_ϕ * sin(ϕ)
    e_y = e_ρ * sin(ϕ) + e_ϕ * cos(ϕ)

    return e_x, e_y, e_z
end
