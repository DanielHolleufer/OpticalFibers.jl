using Test
using OpticalFibers

@testset "Gauss Legendre Pair Symmetry" begin
    weight = OpticalFibers.weight
    gauss_legendre_pairs = OpticalFibers.gauss_legendre_pairs
    gauss_legendre_pairs_positive = OpticalFibers.gauss_legendre_pairs_positive

    n = 500
    pairs1 = gauss_legendre_pairs(2n)
    pairs2 = gauss_legendre_pairs_positive(n)

    @test weight.(pairs1[1:n]) == reverse(weight.(pairs1[n + 1:2n]))
    @test weight.(pairs1[1:n]) == weight.(pairs2[1:n])

    @test cos.(pairs1[1:n]) ≈ -reverse(cos.(pairs1[n + 1:2n])) atol=1e-12
    
    angles1 = [pair.θ for pair in pairs1]
    angles2 = [pair.θ for pair in pairs2]

    @test angles1[1:n] == angles2
end
