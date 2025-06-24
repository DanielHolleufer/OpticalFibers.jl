"""
    AtomTrap

Supertype for atom traps.
"""
abstract type AtomTrap end

"""
    CrossedTweezerTrap

An AtomTrap that contains the parameters of a crossed-beam optical tweezer trap.
"""
struct CrossedTweezerTrap <: AtomTrap
    waist::Float64
    power_per_beam::Float64
    wavelength::Float64
    stark_shifts::StarkShifts
end

function CrossedTweezerTrap(waist::Real, power_per_beam::Real, wavelength::Real,
                            stark_shifts::StarkShifts)
    return CrossedTweezerTrap(Float64(waist), Float64(power_per_beam), Float64(wavelength), 
                              stark_shifts)
end

function Base.show(io::IO, trap::CrossedTweezerTrap)
    println(io, "Crossed beamed tweezer trap parameters:")
    println(io, "Waist = $(trap.waist)")
    println(io, "Power per beam = $(trap.power_per_beam)")
    println(io, "Wavelength = $(trap.wavelength)")
    print(io, trap.stark_shifts)
end

"""
    gaussian_beam_intensity(x::Real, y::Real, z::Real, trap::CrossedTweezerTrap)

Compute the intensity at position (`x`, `y`, `z`) of a gaussian beam parallel to the z-axis
with parameters given in `trap`.
"""
function gaussian_beam_intensity(x::Real, y::Real, z::Real, trap::CrossedTweezerTrap)
    waist = trap.waist
    wavelength = trap.wavelength
    power = trap.power_per_beam
    z_r = π * waist^2 / wavelength
    w = waist * sqrt(1 + (z / z_r)^2)
    intensity_max = 2 * power / (π * waist^2)
    intensity = intensity_max * (waist / w)^2 * exp(-2 * (x^2 + y^2) / w^2)
    return intensity
end

"""
    trap_intensity(x::Real, y::Real, z::Real, trap::CrossedTweezerTrap)

Compute the intensity of the trap at position (`x`, `y`, `z`).
"""
function trap_intensity(x::Real, y::Real, z::Real, trap::CrossedTweezerTrap)
    tweezer_x = gaussian_beam_intensity(z, y, x, trap)
    tweezer_y = gaussian_beam_intensity(x, z, y, trap)
    return tweezer_x + tweezer_y
end

"""
    trap_potential(x::Real, y::Real, z::Real, trap::CrossedTweezerTrap)

Compute the potential of the trap at position (`x`, `y`, `z`).
"""
function trap_potential(x::Real,y::Real, z::Real, trap::CrossedTweezerTrap)
    stark_shift = trap.stark_shifts.ground_state
    return stark_shift * trap_intensity(x, y, z, trap)
end

struct FiberTrap <: AtomTrap
    fiber::Fiber
    polarization::Polarization
    power::Float64
    stark_shifts::StarkShifts
    propagation_index::Int
end

function trap_intensity(x::Real, y::Real, z::Real, trap::FiberTrap)
    ρ = sqrt(x^2 + y^2)
    ϕ = atan(y, x)
    e_x, e_y, e_z = electric_guided_field_cartesian_components(ρ, ϕ, z, 0.0,
                                                               trap.propagation_index, 
                                                               trap.fiber,
                                                               trap.polarization,
                                                               trap.power)
    return abs2(e_x) + abs2(e_y) + abs2(e_z)
end

function trap_potential(x::Real, y::Real, z::Real, trap::FiberTrap)
    stark_shift = trap.stark_shifts.ground_state
    return stark_shift * trap_intensity(x, y, z, trap)
end

struct CrossedTweezerFiberTrap <: AtomTrap
    tweezer_trap::CrossedTweezerTrap
    fiber_trap::FiberTrap
end

function trap_potential(x::Real, y::Real, z::Real, trap::CrossedTweezerFiberTrap)
    tweezer_potential = trap_potential(x, y, z, trap.tweezer_trap)
    fiber_potential = trap_potential(x, y, z, trap.fiber_trap)
    return tweezer_potential + fiber_potential
end
