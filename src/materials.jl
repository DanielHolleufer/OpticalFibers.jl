struct Material{T<:Real}
    B1::T
    B2::T
    B3::T
    C1::T
    C2::T
    C3::T
end

"""
    Material(B1::Real, B2::Real, B3::Real, C1::Real, C2::Real, C3::Real)

Create a Material with Sellmeier coefficients `B₁`, `B₂`, `B₃`, `C₁`, `C₂`, and `C₃`, where
`C₁`, `C₂`, and `C₃` must be given in square micrometers.
"""
function Material(B1::Real, B2::Real, B3::Real, C1::Real, C2::Real, C3::Real)
    return Material(promote(B1, B2, B3, C1, C2, C3)...)
end

function Base.show(io::IO, material::Material)
    println(io, "Material with Sellmeier coefficients:")
    println(io, "B₁ = $(material.B1)")
    println(io, "B₂ = $(material.B2)")
    println(io, "B₃ = $(material.B3)")
    println(io, "C₁ = $(material.C1)μm²")
    println(io, "C₂ = $(material.C2)μm²")
    print(io, "C₃ = $(material.C3)μm²")
end

const SiO2 = Material(0.6961663, 0.4079426, 0.8974794, 0.0684043^2, 0.1162414^2, 9.896161^2)
const BK7 = Material(1.03961212, 0.231792344, 1.01046945, 0.00600069867^2, 0.0200179144^2, 103.560653^2)
const MgF2 = Material(0.48755108, 0.39875031, 2.3120353, 0.001882178, 0.008951888, 566.13559)

"""
    sellmeier_equation(material::Material, wavelength_μm::Real)

Compute the refractive index, `n`, of the material for light with a free space wavelength of
`wavelength_μm`, which must be given in micrometers.

The Sellmeier equation reads

```math
n^{2}(λ) = 1 + \\sum_{i} \\frac{B_{i} λ^{2}}{λ^{2} - C_{i}},
```

where ``B_{i}`` and ``C_{i}`` are constants that depend on the material, and ``λ`` is the
wavelength of light in vacuum. 
"""
function sellmeier_equation(material::Material, wavelength_μm::Real)
    return sqrt(1.0 + material.B1 * wavelength_μm^2 / (wavelength_μm^2 - material.C1)
                + material.B2 * wavelength_μm^2 / (wavelength_μm^2 - material.C2)
                + material.B3 * wavelength_μm^2 / (wavelength_μm^2 - material.C3))
end
