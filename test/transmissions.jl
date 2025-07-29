using DelimitedFiles
using OpticalFibers
using Test

@testset "LinearChains" begin
    a = 0.25
    SiO2 = Material(0.6961663, 0.4079426, 0.8974794, 0.0684043^2, 0.1162414^2, 9.896161^2)
    λ = 0.852
    ω = 2π / λ
    fiber = Fiber(a, λ, SiO2)

    ϕ = 0.0
    z₀ = 0.0
    σx = 0.0
    σy = 0.0
    σz = 0.0
    lattice_constant = 0.1 * λ

    polarization = LinearPolarization(0.0)
    d_dir = [1.0, 0.0, 0.0]
    Γ₀ = 1.0
    d_mag = sqrt(3π * Γ₀ / (ω^3))
    d = d_mag * d_dir
    
    resolution = 2049
    Δs = LinRange(-10Γ₀, 10Γ₀, resolution)

    N_atoms = (5, 10, 20)
    radial_positions = (a + 0.05, a + 0.1, a + 0.2)
    data = Matrix{Float64}(undef, resolution, length(N_atoms) * length(radial_positions))

    for (j, Na) in enumerate(N_atoms)
        for (i, ρ) in enumerate(radial_positions)
            cloud = LinearChain(ρ, ϕ, z₀, σx, σy, σz, lattice_constant, Na, Na)
            positions = atomic_cloud(cloud)

            J_vacuum, Γ_vacuum = vacuum_coefficients(positions, d, ω)
            J_guided, Γ_guided = guided_mode_coefficients(positions, d, fiber)
            J_radiation, Γ_radiation = radiation_mode_coefficients(positions, d, Γ₀, fiber;
                                                                   abstol=1e-6)

            J_total = J_guided + J_vacuum
            Γ_total = Γ_guided + Γ_radiation

            gs = coupling_strengths(d, positions, 1, fiber, polarization)

            ts = transmission_two_level(Δs, fiber, gs, J_total, Γ_total)
            
            data[:, i + (j - 1) * length(radial_positions)] = abs2.(ts)
        end
    end

    test_data = readdlm("test/data/LinearChainTransmission.txt", ',')
    @test data ≈ test_data atol=1e-4
end
