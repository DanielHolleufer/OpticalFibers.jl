using Documenter
using Integrals
using OpticalFibers
using Test

DocMeta.setdocmeta!(OpticalFibers, :DocTestSetup, :(using OpticalFibers); recursive=true)
doctest(OpticalFibers; manual = false)

include("fibers.jl")
include("electric_modes.jl")
include("propagation_constant.jl")
include("gauss_legendre.jl")
include("master_equation_coefficients.jl")
include("transmissions.jl")
