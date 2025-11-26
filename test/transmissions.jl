using DelimitedFiles
using OpticalFibers
using Test
using LinearAlgebra

@testset "LinearChains" begin
    a = 0.25
    SiO2 = Material(0.6961663, 0.4079426, 0.8974794, 0.0684043^2, 0.1162414^2, 9.896161^2)
    λ = 0.852
    ω = 2π / λ
    fiber = Fiber(a, λ, SiO2)
    polarization = LinearPolarization(0.0)
    probe_mode = GuidedMode(fiber, polarization, 1)

    ϕ = 0.0
    z₀ = 0.0
    σx = 0.0
    σy = 0.0
    σz = 0.0
    lattice_constant = 0.1 * λ

    d_dir = [1.0, 0.0, 0.0]
    Γ₀ = 1.0
    d_mag = sqrt(3π * Γ₀ / (ω^3))
    d = d_mag * d_dir

    resolution = 2049
    Δs = LinRange(-10Γ₀, 10Γ₀, resolution)

    NUMBER_OF_ATOMS = (5, 10, 20)
    RADIAL_COORDINATES = (a + 0.05, a + 0.1, a + 0.2)
    N_data = length(NUMBER_OF_ATOMS) * length(RADIAL_COORDINATES)
    data = Matrix{Float64}(undef, resolution, N_data)
    for (j, Na) in enumerate(NUMBER_OF_ATOMS)
        for (i, ρ) in enumerate(RADIAL_COORDINATES)
            cloud = LinearChain(ρ, ϕ, z₀, σx, σy, σz, lattice_constant, a, Na, Na)
            positions = atomic_cloud(cloud)

            J_vacuum, Γ_vacuum = vacuum_coefficients(positions, d, ω)
            J_guided, Γ_guided = guided_coefficients(positions, d, fiber)
            J_radiation, Γ_radiation = radiation_coefficients(positions, d, Γ₀, fiber)

            J_total = J_guided + J_vacuum
            Γ_total = Γ_guided + Γ_radiation

            gs = coupling_strengths(d, positions, probe_mode)

            ts = transmission_two_level(Δs, fiber, gs, J_total, Γ_total)
            
            data[:, i + (j - 1) * length(RADIAL_COORDINATES)] = abs2.(ts)
        end
    end

    file = joinpath(@__DIR__, "data", "LinearChainTransmission.txt")
    test_data = readdlm(file, ',')
    @test data ≈ test_data atol=1e-4
end

@testset "Agreement of Different Setups" begin
    a = 0.05
    λ_probe = 0.399
    ω_probe = 2π / λ_probe
    SiO2 = OpticalFibers.SiO2
    fiber_probe = Fiber(a, λ_probe, SiO2)
    polarization_probe = LinearPolarization(0.0)
    f_probe = 1
    probe = GuidedMode(fiber_probe, polarization_probe, f_probe)
    dβ₀ = propagation_constant_derivative(probe)
    coupling = im * ω_probe * dβ₀ / 2

    d_dir = [1.0, 0.0, 0.0]
    Γ_eg = 2π * 28.0 * 10^6 * SI_to_natural.frequency
    d_mag_eg = sqrt(3π * Γ_eg / (ω_probe^3))
    d_eg = d_mag_eg * d_dir

    λ_control = 0.395
    ω_control = 2π / λ_control
    Ω = 0.0
    control = ExternalField(λ_control, ω_control, Ω)

    Δ_control = 5.0 * Γ_eg
    qₑ = 1.602176634e-19
    a₀ = 5.29177210544e-11
    d_mag_re = 0.005720561291975985 * qₑ * a₀ * SI_to_natural.electric_dipole_moment
    Γ_re = abs2(d_mag_re) * ω_control^3 / (3 * π)
    d_re = d_mag_re * d_dir
    atom = ThreeLevelAtom(Γ_eg, d_eg, Γ_re, d_re, Δ_control)
    shift = Δ_control + im * Γ_re / 2

    positions = [
        1.37067     0.433733    0.304601    0.583608    -1.35302    0.034415    0.67177
        2.45442     -0.0203467  0.381953    0.962057    0.778453    -0.065173   1.08451
        -2.51726    2.64641     8.95915     0.673868    -3.22111    -5.7665     1.54365
    ]

    gs = coupling_strengths(d_eg, positions, probe)
    g_temp1 = similar(gs)
    g_temp2 = similar(gs)
    g_temp3 = similar(gs)
    
    J_guided, Γ_guided = guided_coefficients(positions, d_eg, fiber_probe)
    J_radiation, Γ_radiation = radiation_coefficients(positions, d_eg, Γ_eg, fiber_probe)
    J = J_guided + J_radiation
    Γ = Γ_guided + Γ_radiation

    H = -(J + im * Γ / 2)
    H_hessenberg = hessenberg(H)
    
    detunings = LinRange(-Γ_eg, Γ_eg, 101)
    t1 = Vector{ComplexF64}(undef, length(detunings))
    t2 = Vector{ComplexF64}(undef, length(detunings))
    t3 = Vector{ComplexF64}(undef, length(detunings))

    OpticalFibers.fill_transmissions_three_level!(
        t1, H_hessenberg, g_temp1, detunings, gs, 0.0, coupling, shift, Γ
    )
    OpticalFibers.fill_transmissions_three_level!(
        t2, H, g_temp2, detunings, gs, zeros(7), coupling, shift, Γ
    )
    OpticalFibers.fill_transmissions_two_level!(
        t3, g_temp3, H_hessenberg, detunings, gs, coupling
    )

    @test t1 ≈ t2 atol=1.0e-15
    @test t1 ≈ t3 atol=1.0e-15
end
