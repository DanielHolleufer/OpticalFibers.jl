using DelimitedFiles
using OpticalFibers
using Test

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
    data = Matrix{Float64}(undef, resolution, length(NUMBER_OF_ATOMS) * length(RADIAL_COORDINATES))
    for (j, Na) in enumerate(NUMBER_OF_ATOMS)
        for (i, ρ) in enumerate(RADIAL_COORDINATES)
            cloud = LinearChain(ρ, ϕ, z₀, σx, σy, σz, lattice_constant, a, Na, Na)
            positions = atomic_cloud(cloud)

            J_vacuum, Γ_vacuum = vacuum_coefficients(positions, d, ω)
            J_guided, Γ_guided = guided_mode_coefficients(positions, d, fiber)
            J_radiation, Γ_radiation = OpticalFibers.radiation_mode_coefficients(positions, d, Γ₀, fiber)

            J_total = J_guided + J_vacuum
            Γ_total = Γ_guided + Γ_radiation

            gs = coupling_strengths(d, positions, probe_mode)

            ts = transmission_two_level(Δs, fiber, gs, J_total, Γ_total)
            
            data[:, i + (j - 1) * length(RADIAL_COORDINATES)] = abs2.(ts)
        end
    end

    test_data = readdlm("data/LinearChainTransmission.txt", ',')
    @test data ≈ test_data atol=1e-4
end
