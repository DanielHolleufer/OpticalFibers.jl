using Documenter
using Integrals
using OpticalFibers
using Test

DocMeta.setdocmeta!(OpticalFibers, :DocTestSetup, :(using OpticalFibers); recursive=true)
doctest(OpticalFibers; manual = false)

include("fibers.jl")
