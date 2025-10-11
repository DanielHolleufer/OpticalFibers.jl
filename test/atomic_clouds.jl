using OpticalFibers
using Test

@testset "Gaussian Clouds" begin
    σ_x = 1.0
    σ_y = 1.0
    σ_z = 1.0
    fiber_radius = 0.1
    exclusion_zone = 0.0
    peak_density = 1.0
    cloud = GaussianCloud(σ_x, σ_y, σ_z, fiber_radius, exclusion_zone, peak_density)
    N = cloud.number_of_atoms
    peak_density_approximation = OpticalFibers.peak_density_gaussian_cloud(σ_x, σ_y, σ_z, fiber_radius, exclusion_zone, N)
    @test peak_density ≈ peak_density_approximation atol=0.1

    σ_x = 5.0
    σ_y = 5.0
    σ_z = 10.0
    fiber_radius = 0.05
    exclusion_zone = 0.1
    peak_density = 10.0
    cloud = GaussianCloud(σ_x, σ_y, σ_z, fiber_radius, exclusion_zone, peak_density)
    N = cloud.number_of_atoms
    peak_density_approximation = OpticalFibers.peak_density_gaussian_cloud(σ_x, σ_y, σ_z, fiber_radius, exclusion_zone, N)
    @test peak_density ≈ peak_density_approximation atol=0.1
end
