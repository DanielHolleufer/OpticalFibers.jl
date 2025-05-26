using DelimitedFiles
using OpticalFibers

@testset "Single Two-level Atom" begin
    a = 0.25
    λ = 0.852
    SiO2 = Material(0.6961663, 0.4079426, 0.8974794, 0.0684043^2, 0.1162414^2, 9.896161^2)
    fiber = Fiber(a, λ, SiO2)

    
end
