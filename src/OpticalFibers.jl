module OpticalFibers

export Material
export Fiber, PolarizationBasis, LinearPolarization, CircularPolarization, radius,
    wavelength, frequency, material, refractive_index, propagation_constant,
    propagation_constant_derivative, normalized_frequency, effective_refractive_index
export electric_guided_mode_cylindrical_base_components,
    electric_guided_mode_profile_cartesian_components,
    electric_guided_field_cartesian_components, electric_guided_field_cartesian_vector
export vacuum_coefficients, guided_mode_coefficients, guided_mode_directional_coefficients,
    radiation_mode_coefficients, radiation_mode_decay_coefficients,
    radiation_mode_directional_coefficients
export single_two_level_transmission, single_three_level_transmission, optical_depth,
    coupling_strengths, transmission_three_level, transmission_two_level
export gaussian_atomic_cloud, tweezer_trap_intensity, tweezer_trap_potential,
    fiber_potential, full_potential, gaussian_beam_intensity

include("materials.jl")
include("fibers.jl")
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
using SpecialFunctions: gamma, gamma_inc

end
