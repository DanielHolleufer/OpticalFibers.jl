struct Fiber{T<:Real}
    radius::T
    wavelength::T
    frequency::T
    material::Material{T}
    refractive_index::T
    propagation_constant::T
    propagation_constant_derivative::T
    internal_parameter::T
    external_parameter::T
    s_parameter::T
    normalization_constant::T
    function Fiber(radius::T, wavelength::T, material::Material{T}) where {T<:Real}
        radius < 0 && throw(DomainError(radius, "The fiber radius must be greater than zero."))
        wavelength < 0 && throw(DomainError(wavelength, "The wavelength must be greater than zero."))

        ω = 2π / wavelength
        n = sellmeier_equation(material, wavelength)

        V = _normalized_frequency(radius, n, ω)
        V > 2.40482 && @warn "Fiber supports multiple modes."

        β = _propagation_constant(radius, n, ω)
        dβ = _propagation_constant_derivative(radius, n, ω)
        h = sqrt(n^2 * ω^2 - β^2)
        q = sqrt(β^2 - ω^2)
        J1 = besselj1(h * radius)
        K1 = besselk1(q * radius)
        dJ1 = 1 / 2 * (besselj0(h * radius) - besselj(2, h * radius))
        dK1 =  -1 / 2 * (besselk0(q * radius) + besselk(2, q * radius))
        s_numerator = (1 / (h^2 * radius^2) + 1 / (q^2 * radius^2))
        s_denominator = (dJ1 / (h * radius * J1) + dK1 / (q * radius * K1))
        s = s_numerator / s_denominator
        C = guided_mode_normalization_constant(radius, n, β, h, q, s)

        return new{T}(radius, wavelength, ω, material, n, β, dβ, h, q, s, C)
    end
end

function Base.show(io::IO, fiber::Fiber)
    println(io, "Optical fiber with parameters:")
    println(io, "Radius: $(fiber.radius)μm")
    println(io, "Wavelength: $(fiber.wavelength)μm")
    println(io, "Frequency: 2π • $(fiber.frequency / (2π))")
    println(io, "Refractive index: $(fiber.refractive_index)")
    println(io, "Propagation constant: $(fiber.propagation_constant)")
    print(io, fiber.material)
end

"""
    radius(fiber::Fiber)

Return the radius of the fiber in micrometers.

# Examples
```jldoctest
julia> fiber = Fiber(0.1, 0.4, OpticalFibers.SiO2);

julia> radius(fiber)
0.1
```
"""
radius(fiber::Fiber) = fiber.radius

"""
    wavelength(fiber::Fiber)

Return the wavelength of the fiber mode in micrometers.

# Examples
```jldoctest
julia> fiber = Fiber(0.1, 0.4, OpticalFibers.SiO2);

julia> wavelength(fiber)
0.4
```
"""
wavelength(fiber::Fiber) = fiber.wavelength

"""
    frequency(fiber::Fiber)

Return the angular frequency of the fiber mode in units of inverse micrometers with the
speed of light `c = 1`.

# Examples
```jldoctest
julia> fiber = Fiber(0.1, 0.4, OpticalFibers.SiO2);

julia> frequency(fiber)
15.707963267948966
```
"""
frequency(fiber::Fiber) = fiber.frequency

"""
    material(fiber::Fiber)

Return the material of the fiber.

# Examples
```jldoctest
julia> fiber = Fiber(0.1, 0.4, OpticalFibers.SiO2);

julia> material(fiber)
Material with Sellmeier coefficients:
B₁ = 0.6961663
B₂ = 0.4079426
B₃ = 0.8974794
C₁ = 0.00467914825849μm²
C₂ = 0.013512063073959999μm²
C₃ = 97.93400253792099μm²
```
"""
material(fiber::Fiber) = fiber.material

"""
    refractive_index(fiber::Fiber)

Return the refractive index of the fiber for light with the same wavelength as the fiber
mode.

# Examples
```jldoctest
julia> fiber = Fiber(0.1, 0.4, OpticalFibers.SiO2);

julia> refractive_index(fiber)
1.4701161185594052
```
"""
refractive_index(fiber::Fiber) = fiber.refractive_index

"""
    propagation_constant(fiber::Fiber)

Return the propagation constant of the fiber in inverse micrometers for light with the same
wavelength as the fiber mode.

# Examples
```jldoctest
julia> fiber = Fiber(0.1, 0.4, OpticalFibers.SiO2);

julia> propagation_constant(fiber)
17.242890161273014
```
"""
propagation_constant(fiber::Fiber) = fiber.propagation_constant

"""
    propagation_constant_derivative(fiber::Fiber)

Return the derivative of the propagation constant with respect to angular frequency of the
fiber evaluated at the angular frequency of the fiber mode.

# Examples
```jldoctest
julia> fiber = Fiber(0.1, 0.4, OpticalFibers.SiO2);

julia> propagation_constant_derivative(fiber)
1.4431087436150847
```
"""
propagation_constant_derivative(fiber::Fiber) = fiber.propagation_constant_derivative

"""
    group_velocity(fiber::Fiber)

Return the group velocity of the fiber mode in units of the speed of light `c = 1`.

# Examples
```jldoctest
julia> fiber = Fiber(0.1, 0.4, OpticalFibers.SiO2);

julia> group_velocity(fiber)
0.6929484728191256
```
"""
group_velocity(fiber) = inv(propagation_constant_derivative(fiber))

"""
    normalized_frequency(fiber::Fiber)

Return the normalized frequency of the fiber mode.

# Examples
```jldoctest
julia> fiber = Fiber(0.1, 0.4, OpticalFibers.SiO2);

julia> normalized_frequency(fiber)
1.692704437607302
```
"""
function normalized_frequency(fiber::Fiber)
    a = radius(fiber)
    n = refractive_index(fiber)
    ω = frequency(fiber)
    return _normalized_frequency(a, n, ω)
end

_normalized_frequency(a::Real, n::Real, ω::Real) = ω * a * sqrt(n^2 - 1)

"""
    effective_refractive_index(fiber::Fiber)

Return the effective refractive index of the fiber.

# Examples
```jldoctest
julia> fiber = Fiber(0.1, 0.4, OpticalFibers.SiO2);

julia> effective_refractive_index(fiber)
1.0977164809428834
```
"""
function effective_refractive_index(fiber::Fiber)
    λ = wavelength(fiber)
    β = propagation_constant(fiber)
    return β * λ / 2π
end

"""
    guided_mode_normalization_constant(a::Real, n::Real, β::Real, h::Real, q::Real, s::Real)

Compute the normalization constant of an electric guided fiber mode.

The fiber modes are normalized according to the condition
```math
\\int_{0}^{\\infty} \\! \\mathrm{d} \\rho \\int_{0}^{2 \\pi} \\! \\mathrm{d} \\phi \\, 
n^{2}(\\rho) \\lvert \\mathrm{\\mathbf{e}}(\\rho, \\phi) \\rvert^{2} = 1,
```
where ``n(\\rho)`` is the step index refractive index given as
```math
n(\\rho) = \\begin{cases}
    n, & \\rho < a, \\\\
    1 & \\rho > a,
\\end{cases}
```
and
``\\mathrm{\\mathbf{e}}(\\rho, \\phi) = e_{\\rho} \\hat{\\mathrm{\\mathbf{\\rho}}} \
+ e_{\\phi} \\hat{\\mathrm{\\mathbf{\\phi}}} + e_{z} \\hat{\\mathrm{\\mathbf{z}}}``,
where the components are given by [`electric_guided_mode_base`](@ref).
"""
function guided_mode_normalization_constant(
    a::Real, n::Real, β::Real, h::Real, q::Real, s::Real
)
    j0 = besselj0(h * a)
    j1 = besselj1(h * a)
    j2 = besselj(2, h * a)
    j3 = besselj(3, h * a)
    k0 = besselk0(q * a)
    k1 = besselk1(q * a)
    k2 = besselk(2, q * a)
    k3 = besselk(3, q * a)
    
    C_in_1 = (1 - s)^2 * (j0^2 + j1^2) 
    C_in_2 = (1 + s)^2 * (j2^2 - j1 * j3) 
    C_in_3 = 2 * (h / β)^2 * (j1^2 - j0 * j2) 
    C_in = (n * q * k1 / (h * j1))^2 * (C_in_1 + C_in_2 + C_in_3)

    C_out_1 = (1 - s)^2 * (k0^2 - k1^2)
    C_out_2 = (1 + s)^2 * (k2^2 - k1 * k3)
    C_out_3 = 2 * (q / β)^2 * (k1^2 - k0 * k2)
    C_out = -(C_out_1 + C_out_2 + C_out_3)

    return 1 / sqrt(2π * (C_in + C_out)) * 1 / a
end
