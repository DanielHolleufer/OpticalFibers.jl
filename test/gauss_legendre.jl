using Test
using OpticalFibers

@testset "Gauss Legendre Pair Symmetry" begin
    gauss_legendre_pairs = OpticalFibers.gauss_legendre_pairs

    n = 500
    angles, weights = gauss_legendre_pairs(n)

    @test weights[1:250] == reverse(weights[251:end])
end
