"""
    FiberMaterial(B1::Real, B2::Real, B3::Real, C1::Real, C2::Real, C3::Real)

Create a FiberMaterial with Sellmeier coefficients 'B₁', 'B₂', 'B₃', 'C₁', 'C₂', and 'C₃',
where 'C₁', 'C₂', and 'C₃' must be given in square micrometers.
"""
struct Material{T<:Real}
    B1::T
    B2::T
    B3::T
    C1::T
    C2::T
    C3::T
end

function Material(B1::Real, B2::Real, B3::Real, C1::Real, C2::Real, C3::Real)
    return Material(promote(B1, B2, B3, C1, C2, C3)...)
end

function Base.show(io::IO, material::Material)
    println("Material with Sellmeier coefficients:")
    println(io, "B₁ = $(material.B1)")
    println(io, "B₂ = $(material.B2)")
    println(io, "B₃ = $(material.B3)")
    println(io, "C₁ = $(material.C1)μm²")
    println(io, "C₂ = $(material.C2)μm²")
    print(io, "C₃ = $(material.C3)μm²")
end

"""
    sellmeier_equation(material, λ)

Compute the refractive index, 'n', of the material for light with a free space wavelength of
`λ`, which must be given in micrometers.

The Sellmeier equation reads
``n^{2}(λ) = 1 + \\sum_{i} \\frac{B_{i} λ^{2}}{λ^{2} - C_{i}},``
where 'B_{i}' and 'C_{i}' are constants that depends on the material. 
"""
function sellmeier_equation(material::Material, λ::Real)
    return sqrt(1 + material.B1 * λ^2 / (λ^2 - material.C1)
                + material.B2 * λ^2 / (λ^2 - material.C2)
                + material.B3 * λ^2 / (λ^2 - material.C3))
end
