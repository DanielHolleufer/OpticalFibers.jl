module OpticalFibers

export Material
export Fiber, PolarizationBasis, LinearPolarization, CircularPolarization
export vacuum_coefficients, guided_mode_coefficients, guided_mode_directional_coefficients,
    radiation_mode_coefficients, radiation_mode_decay_coefficients,
    radiation_mode_directional_coefficients

include("materials.jl")
include("propagation_constant.jl")
include("electric_modes.jl")
include("master_equation_coefficients.jl")
include("atomic_cloud.jl")

using Bessels
using Distributions
using Integrals
using LinearAlgebra
using NonlinearSolve
using Random

end
