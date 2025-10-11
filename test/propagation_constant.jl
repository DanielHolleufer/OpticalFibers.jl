using OpticalFibers

@testset "Propagation Constant" begin
    # Determination of the propagation constant, β, is tested in the single-mode regime. 
    # β values to test against are obtained from:
    # https://www.computational-photonics.eu/fims.html

    λ = 0.2
    a = 0.1
    n = 1.2
    k = 2π / λ
    β = OpticalFibers._propagation_constant(a, n, k)
    @test β ≈ 34.00910381 atol = 1e-6

    λ = 0.987
    a = 0.123
    n = 1.5
    k = 2π / λ
    β = OpticalFibers._propagation_constant(a, n, k)
    @test β ≈ 6.369099651 atol = 1e-6

    λ = 0.395
    a = 0.05
    SiO2 = Material(0.6961663, 0.4079426, 0.8974794, 0.0684043^2, 0.1162414^2, 9.896161^2)
    n = OpticalFibers.sellmeier_equation(SiO2, λ)
    k = 2π / λ
    β = OpticalFibers._propagation_constant(a, n, k)
    @test β ≈ 15.91342987 atol = 1e-6

    # This test uses the parameters found in https://arxiv.org/abs/2407.02278
    λ = 0.852
    a = 0.2
    n = 1.45
    k = 2π / λ
    β = OpticalFibers._propagation_constant(a, n, k)
    @test β ≈ 7.875273003 atol = 1e-6
end