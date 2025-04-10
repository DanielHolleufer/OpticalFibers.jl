using DelimitedFiles
using OpticalFibers
using Test

@testset "Guided Mode Coefficients" begin
    # We test if our code agrees with the results from
    # https://doi.org/10.1103/PhysRevA.90.023805
    
    d = [1.0im, 0.0, -1.0] / sqrt(2)
    λ = 0.852
    a = 0.250
    ω = 2π / λ
    SiO2 = Material(0.6961663, 0.4079426, 0.8974794, 0.0684043^2, 0.1162414^2, 9.896161^2)
    polarization_basis = CircularPolarization()
    fiber = Fiber(a, λ, SiO2)

    ρs = LinRange(0.0, 1.0, 128)
    zs = LinRange(0.0, 2.0, 128)

    Γ_22_ρs = zeros(ComplexF64, length(ρs))
    J_12_ρs = zeros(ComplexF64, length(ρs))
    Γ_12_ρs = zeros(ComplexF64, length(ρs))
    Γ_22_p_ρs = zeros(ComplexF64, length(ρs))
    Γ_12_p_ρs = zeros(ComplexF64, length(ρs))
    Γ_22_m_ρs = zeros(ComplexF64, length(ρs))
    Γ_12_m_ρs = zeros(ComplexF64, length(ρs))
    for i in eachindex(ρs)
        r = [[a, 0.0, 0.0] [a + eps(a) + ρs[i], 0.0, eps()]]
        J, Γ = guided_mode_coefficients(r, d, fiber, polarization_basis)
        _, Γ_p = guided_mode_directional_coefficients(r, d, fiber, 1, polarization_basis)
        _, Γ_m = guided_mode_directional_coefficients(r, d, fiber, -1, polarization_basis)
        Γ_22_ρs[i] = Γ[2, 2]
        J_12_ρs[i] = J[1, 2]
        Γ_12_ρs[i] = Γ[1, 2]
        Γ_22_p_ρs[i] = Γ_p[2, 2]
        Γ_12_p_ρs[i] = Γ_p[1, 2]
        Γ_22_m_ρs[i] = Γ_m[2, 2]
        Γ_12_m_ρs[i] = Γ_m[1, 2]
    end

    Γ_22_zs = zeros(ComplexF64, length(zs))
    J_12_zs = zeros(ComplexF64, length(zs))
    Γ_12_zs = zeros(ComplexF64, length(zs))
    Γ_22_p_zs = zeros(ComplexF64, length(zs))
    Γ_12_p_zs = zeros(ComplexF64, length(zs))
    Γ_22_m_zs = zeros(ComplexF64, length(zs))
    Γ_12_m_zs = zeros(ComplexF64, length(zs))
    for i in eachindex(zs)
        r = [[a, 0.0, 0.0] [a, 0.0, eps() + zs[i]]]
        J, Γ = guided_mode_coefficients(r, d, fiber, polarization_basis)
        _, Γ_p = guided_mode_directional_coefficients(r, d, fiber, 1, polarization_basis)
        _, Γ_m = guided_mode_directional_coefficients(r, d, fiber, -1, polarization_basis)
        Γ_22_zs[i] = Γ[2, 2]
        J_12_zs[i] = J[1, 2]
        Γ_12_zs[i] = Γ[1, 2]
        Γ_22_p_zs[i] = Γ_p[2, 2]
        Γ_12_p_zs[i] = Γ_p[1, 2]
        Γ_22_m_zs[i] = Γ_m[2, 2]
        Γ_12_m_zs[i] = Γ_m[1, 2]
    end
    
    # Load data that when plotted shows excellent agreement with
    # https://doi.org/10.1103/PhysRevA.90.023805
    test_data = readdlm("data/GuidedModeMasterEquationCoefficients.txt", ',', ComplexF64)
    
    # Store results in the same format as the test data
    data = [Γ_22_ρs J_12_ρs Γ_12_ρs Γ_22_p_ρs Γ_12_p_ρs Γ_22_m_ρs Γ_12_m_ρs Γ_22_zs J_12_zs Γ_12_zs Γ_22_p_zs Γ_12_p_zs Γ_22_m_zs Γ_12_m_zs]
    @test data ≈ test_data atol=1e-6
end
