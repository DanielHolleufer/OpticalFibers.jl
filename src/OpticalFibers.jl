module OpticalFibers

export Material
export Fiber, PolarizationBasis, LinearPolarization, CircularPolarization
export vacuum_coefficients, guided_mode_coefficients, guided_mode_directional_coefficients,
    radiation_mode_coefficients, radiation_mode_decay_coefficients,
    radiation_mode_directional_coefficients
export single_two_level_transmission, single_three_level_transmission, optical_depth,
    coupling_strengths, transmission_three_level

include("materials.jl")
include("propagation_constant.jl")
include("electric_modes.jl")
include("master_equation_coefficients.jl")
include("atomic_cloud.jl")
include("transmissions.jl")

using Bessels
using Distributions
using Integrals
using LinearAlgebra
using NonlinearSolve
using Optim
using Random

end
