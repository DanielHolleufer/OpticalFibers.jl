using OpticalFibers
using Test

@testset "Natural Constants" begin
    # Test that the natural constant c, ħ, k_B, ϵ₀, and μ₀ all equal 1 in natural units.
    c = 2.99792458e8
    ħ = 6.62607015e-34 / 2π
    k_B = 1.380649e-23
    ε₀ = 8.8541878188e-12
    μ₀ = 1.0 / (ε₀ * c^2)

    SI_to_natural = OpticalFibers.SI_to_natural

    @test c * SI_to_natural.velocity ≈ 1 atol=1e-15
    @test ħ * SI_to_natural.energy * SI_to_natural.time ≈ 1 atol=1e-15
    @test k_B * SI_to_natural.energy / SI_to_natural.temperature ≈ 1 atol=1e-15
    @test ε₀ * SI_to_natural.permitivity ≈ 1 atol=1e-15
    @test μ₀ * SI_to_natural.permeability ≈ 1 atol=1e-15

    # Also test that computing the fine structure constant in both systems give the same
    # dimensionless value.
    α = 0.007297352564

    qₑ_SI = 1.602176634e-19
    α_SI = qₑ_SI^2 / (4π * ε₀ * ħ * c)

    qₑ_natural = qₑ_SI * SI_to_natural.charge
    α_natural = qₑ_natural^2 / (4π)

    @test α ≈ α_SI atol=1e-12
    @test α ≈ α_natural atol=1e-12
end

@testset "Inverse Conversions" begin
    # Test that the conversion factors from SI to natural and from natural to SI are each
    # others inverses.
    SI_to_natural = OpticalFibers.SI_to_natural
    natural_to_SI = OpticalFibers.natural_to_SI

    @test SI_to_natural.length * natural_to_SI.length ≈ 1 atol=1e-15
    @test SI_to_natural.time * natural_to_SI.time ≈ 1 atol=1e-15
    @test SI_to_natural.mass * natural_to_SI.mass ≈ 1 atol=1e-15
    @test SI_to_natural.current * natural_to_SI.current ≈ 1 atol=1e-15
    @test SI_to_natural.temperature * natural_to_SI.temperature ≈ 1 atol=1e-15
    @test SI_to_natural.area * natural_to_SI.area ≈ 1 atol=1e-15
    @test SI_to_natural.volume * natural_to_SI.volume ≈ 1 atol=1e-15
    @test SI_to_natural.frequency * natural_to_SI.frequency ≈ 1 atol=1e-15
    @test SI_to_natural.velocity * natural_to_SI.velocity ≈ 1 atol=1e-15
    @test SI_to_natural.acceleration * natural_to_SI.acceleration ≈ 1 atol=1e-15
    @test SI_to_natural.momentum * natural_to_SI.momentum ≈ 1 atol=1e-15
    @test SI_to_natural.force * natural_to_SI.force ≈ 1 atol=1e-15
    @test SI_to_natural.energy * natural_to_SI.energy ≈ 1 atol=1e-15
    @test SI_to_natural.power * natural_to_SI.power ≈ 1 atol=1e-15
    @test SI_to_natural.intensity * natural_to_SI.intensity ≈ 1 atol=1e-15
    @test SI_to_natural.charge * natural_to_SI.charge ≈ 1 atol=1e-15
    @test SI_to_natural.electric_field * natural_to_SI.electric_field ≈ 1 atol=1e-15
    @test SI_to_natural.electric_dipole_moment * natural_to_SI.electric_dipole_moment ≈ 1 atol=1e-15
    @test SI_to_natural.capacitance * natural_to_SI.capacitance ≈ 1 atol=1e-15
    @test SI_to_natural.permitivity * natural_to_SI.permitivity ≈ 1 atol=1e-15
    @test SI_to_natural.permeability * natural_to_SI.permeability ≈ 1 atol=1e-15
end

@testset "Derived Units" begin
    # Test that computing physical quantities in SI units and then converting them to
    # natural units is equivalent to computing them directly in natural units.
    c = 2.99792458e8
    ħ = 1.0545718e-34
    ε₀ = 8.854187817e-12

    SI_to_natural = OpticalFibers.SI_to_natural

    λ_SI = 0.0000004
    ω_SI = 2π * c / λ_SI
    Γ_SI = 2π * 28.0 * 10^6
    d_SI = sqrt(3π * ħ * ε₀ * c^3 * Γ_SI / (ω_SI^3))

    λ_natural = 0.4
    ω_natural = 2π / λ_natural
    Γ_natural = 2π * 28.0 * 10^6 * SI_to_natural.frequency
    d_natural = sqrt(3π * Γ_natural / (ω_natural^3))

    @test ω_SI * SI_to_natural.frequency ≈ ω_natural atol=1e-12
    @test d_SI * SI_to_natural.electric_dipole_moment ≈ d_natural atol=1e-12

    @test SI_to_natural.length * SI_to_natural.charge == SI_to_natural.electric_dipole_moment
end

@testset "Unit Conversion Reproduceability" begin
    # Test that the constant structure containing the conversion factors matches with what
    # the UnitConversions constructor outputs when given SI values for c, ħ, k_B, and ϵ₀.
    meter = 1.0e6
    c = 2.99792458e8
    ħ = 6.62607015e-34 / 2π
    k_B = 1.380649e-23
    ε₀ = 8.8541878188e-12
    SI_to_natural = OpticalFibers.SI_to_natural
    @test OpticalFibers.UnitConversions(meter, c, ħ, k_B, ε₀) == SI_to_natural
end
