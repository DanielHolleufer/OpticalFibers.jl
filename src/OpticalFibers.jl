module OpticalFibers

export Material
export Fiber, radius, wavelength, frequency, material, refractive_index,
    propagation_constant, propagation_constant_derivative, normalized_frequency,
    effective_refractive_index
export Polarization, polarization, LinearPolarization, CircularPolarization, GuidedMode,
    GuidedField, direction, ExternalField, electric_guided_mode_base,
    electric_guided_mode_profile_cylindrical, electric_guided_mode_profile_cartesian,
    electric_guided_field_cartesian
export ThreeLevelAtom, StarkShifts, ytterbium_stark_shift_1070nm_natural_units, 
    ytterbium_stark_shift_395nm_natural_units
export AtomTrap, CrossedTweezerTrap, FiberTrap, CrossedTweezerFiberTrap, trap_intensity,
    trap_potential
export AtomicCloud, atomic_density_distribution, GaussianCloud, atomic_density_distribution,
    atomic_cloud, CrossedTweezerFiberTrappedCloud, LinearChain
export vacuum_coefficients, guided_mode_coefficients, guided_mode_directional_coefficients,
    radiation_mode_coefficients, radiation_mode_decay_coefficients,
    radiation_mode_directional_coefficients
export single_two_level_transmission, single_three_level_transmission, optical_depth,
    coupling_strengths, transmission_three_level, transmission_two_level,
    probe_detuning_range
export transmission_coefficient_continuous_propagation
export GaussLegendrePair, gauss_legendre_pair, gauss_legendre_pairs

include("materials.jl")
include("fibers.jl")
include("electric_modes.jl")
include("atomic_structures.jl")
include("atom_traps.jl")
include("atomic_cloud.jl")
include("propagation_constant.jl")
include("master_equation_coefficients.jl")
include("transmissions.jl")
include("continuous_propagation.jl")
include("gauss_legendre.jl")

using Bessels
using Cuba
using Distributions
using Integrals
using LinearAlgebra
using NonlinearSolve
using Optim
using Optimization
using OptimizationNLopt
using Random
using SpecialFunctions: gamma, gamma_inc

end
