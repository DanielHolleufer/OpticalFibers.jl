using DelimitedFiles
using OpticalFibers
using Test

@testset "Vacuum Coefficients" begin
    # We test our code against the results from
    # https://doi.org/10.1103/PhysRevA.90.023805

    d = [1.0im, 0.0, -1.0] / sqrt(2)
    λ = 0.852
    a = 0.250
    ω = 2π / λ
    SiO2 = Material(0.6961663, 0.4079426, 0.8974794, 0.0684043^2, 0.1162414^2, 9.896161^2)
    fiber = Fiber(a, λ, SiO2)
    Γ₀ = sum(abs2, d) * ω^3 / 3π

    zs = LinRange(0.0, 4.0, 128)
    J_12_zs = zeros(ComplexF64, length(zs))

    for i in eachindex(zs)
        r = [[a, 0.0, 0.0] [a, 0.0, eps() + zs[i]]]
        J, _ = vacuum_coefficients(r, d, ω)
        J_12_zs[i] = J[1, 2]
    end

    # We also test our code against the results from
    # https://doi.org/10.22331/q-2023-08-22-1091
    dx = [1.0, 0.0, 0.0]
    dz = [0.0, 0.0, 1.0]

    zs_2 = LinRange(0.0, λ, 128)
    J_12_zs_x = zeros(ComplexF64, length(zs))
    J_12_zs_z = zeros(ComplexF64, length(zs))
    for i in eachindex(zs)
        r = [[a, 0.0, 0.0] [a, 0.0, eps() + zs_2[i]]]
        Jx, _ = vacuum_coefficients(r, dx, ω)
        Jz, _ = vacuum_coefficients(r, dz, ω)
        J_12_zs_x[i] = Jx[1, 2]
        J_12_zs_z[i] = Jz[1, 2]
    end

    # Load data that when plotted shows excellent agreement with
    # https://doi.org/10.1103/PhysRevA.90.023805 and
    # https://doi.org/10.22331/q-2023-08-22-1091
    # The data has been created with the sign convention from
    # https://doi.org/10.22331/q-2023-08-22-1091
    test_data = readdlm("data/VacuumMasterEquationCoefficients.txt", ',', ComplexF64)
    data = [J_12_zs J_12_zs_x J_12_zs_z]
    @test data ≈ test_data atol=1e-6
end

@testset "Guided Mode Coefficients" begin
    # We test our code against the results from
    # https://doi.org/10.1103/PhysRevA.90.023805
    
    d = [1.0im, 0.0, -1.0] / sqrt(2)
    λ = 0.852
    a = 0.250
    ω = 2π / λ
    SiO2 = Material(0.6961663, 0.4079426, 0.8974794, 0.0684043^2, 0.1162414^2, 9.896161^2)
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
        J, Γ = guided_mode_coefficients(r, d, fiber)
        _, Γ₊, _, Γ₋ = guided_mode_directional_coefficients(r, d, fiber)
        Γ_22_ρs[i] = Γ[2, 2]
        J_12_ρs[i] = J[1, 2]
        Γ_12_ρs[i] = Γ[1, 2]
        Γ_22_p_ρs[i] = Γ₊[2, 2]
        Γ_12_p_ρs[i] = Γ₊[1, 2]
        Γ_22_m_ρs[i] = Γ₋[2, 2]
        Γ_12_m_ρs[i] = Γ₋[1, 2]
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
        J, Γ = guided_mode_coefficients(r, d, fiber)
        _, Γ₊, _, Γ₋ = guided_mode_directional_coefficients(r, d, fiber)
        Γ_22_zs[i] = Γ[2, 2]
        J_12_zs[i] = J[1, 2]
        Γ_12_zs[i] = Γ[1, 2]
        Γ_22_p_zs[i] = Γ₊[2, 2]
        Γ_12_p_zs[i] = Γ₊[1, 2]
        Γ_22_m_zs[i] = Γ₋[2, 2]
        Γ_12_m_zs[i] = Γ₋[1, 2]
    end
    
    # Load data that when plotted shows excellent agreement with
    # https://doi.org/10.1103/PhysRevA.90.023805
    test_data = readdlm("data/GuidedModeMasterEquationCoefficients.txt", ',', ComplexF64)
    
    # Store results in the same format as the test data. Note that we have a different sign
    # convention for the dipole-dipole interaction coeffecients, so we multiply ours by -1.
    data = [Γ_22_ρs -J_12_ρs Γ_12_ρs Γ_22_p_ρs Γ_12_p_ρs Γ_22_m_ρs Γ_12_m_ρs Γ_22_zs -J_12_zs Γ_12_zs Γ_22_p_zs Γ_12_p_zs Γ_22_m_zs Γ_12_m_zs]
    @test data ≈ test_data atol=1e-6
end
