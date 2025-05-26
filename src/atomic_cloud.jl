function gaussian_cloud_distribution(x, y, z, std_x, std_y, std_z, fiber, exclusion_zone = 0.0)
    if sqrt(x^2 + y^2) ≤ fiber.radius + exclusion_zone
        return 0.0
    else
        return exp(-x^2 / (2 * std_x^2) - y^2 / (2 * std_y^2) - z^2 / (2 * std_z^2))
    end
end

function gaussian_cloud_normalization_constant(std_x, std_y, std_z, fiber, exclusion_zone = 0.0)
    p = std_x, std_y, std_z, fiber, exclusion_zone
    domain = ([-5 * std_x, -5 * std_y, -5 * std_z], [5 * std_x, 5 * std_y, 5 * std_z])
    prob = IntegralProblem(gaussian_cloud_normalization_constant_integrand, domain, p)
    sol = solve(prob, CubaDivonne())
    return sol.u
end

function gaussian_cloud_normalization_constant_integrand(u, p)
    std_x, std_y, std_z, fiber, exclusion_zone = p
    return gaussian_cloud_distribution(u[1], u[2], u[3], std_x, std_y, std_z, fiber, exclusion_zone)
end

function gaussian_atomic_cloud(N, distribution, fiber, exclusion_zone = 0.0)
    r = zeros(3, N)
    for i in 1:N
        while sqrt(r[1, i]^2 + r[2, i]^2) ≤ fiber.radius + exclusion_zone
            r[:, i] = rand(distribution)
        end
    end
    return r
end

function atom_number_gaussian(peak_density_μm, fiber_control, std_x, std_y, std_z, exclusion_zone)
    a = fiber_control.radius
    p = std_x, std_y, std_z, fiber_control, exclusion_zone
    normalization_constant = OpticalFibers.gaussian_cloud_normalization_constant(std_x, std_y, std_z, fiber_control, exclusion_zone)
    P_M = OpticalFibers.maximum_within_box(gaussian_cloud_normalization_constant_integrand, [2a, 2a, 2a], 5 * std_x, p; algorithm=LBFGS()) / normalization_constant
    return round(Int, peak_density_μm / P_M)
end

struct CrossedTweezerTrap
    waist::Float64
    power_per_beam::Float64
    wavelength::Float64
    stark_shifts::StarkShifts
end

"""
    gaussian_beam_intensity(x::Real, y::Real, z::Real, waist::Real, power::Real, wavelength::Real)

Compute the intensity at position (`x`, `y`, `z`) of a gaussian beam parallel to the z-axis
with parameters `waist`, `power`, and `wavelength`.
"""
function gaussian_beam_intensity(x::Real, y::Real, z::Real, tweezer_trap::CrossedTweezerTrap)
    waist = tweezer_trap.waist
    wavelength = tweezer_trap.wavelength
    power = tweezer_trap.power_per_beam
    z_r = π * waist^2 / wavelength
    w = waist * sqrt(1 + (z / z_r)^2)
    intensity_max = 2 * power / (π * waist^2)
    intensity = intensity_max * (waist / w)^2 * exp(-2 * (x^2 + y^2) / w^2)
    return intensity
end

"""
    tweezer_trap_intensity(x::Real, y::Real, z::Real, waist::Real, power::Real, wavelength::Real)

Compute the intensity at position (`x`, `y`, `z`) of a dipole tweezer trap consisting of two
beam parallel to the gaussian beams, one along the x-axis and one along the y-axis, where
both beams have parameters `waist`, `power`, and `wavelength`.
"""
function tweezer_trap_intensity(x::Real, y::Real, z::Real,tweezer_trap::CrossedTweezerTrap)
    tweezer_x = gaussian_beam_intensity(z, y, x, tweezer_trap)
    tweezer_y = gaussian_beam_intensity(x, z, y, tweezer_trap)
    return tweezer_x + tweezer_y
end

"""
    tweezer_trap_potential(x::Real, y::Real, z::Real, waist::Real, power::Real, wavelength::Real, stark_shift::Real)

Compute the potential at position (`x`, `y`, `z`) of a dipole tweezer trap consisting of two
beam parallel to the gaussian beams, one along the x-axis and one along the y-axis, where
both beams have parameters `waist`, `power`, and `wavelength`, and the stark_shift per unit
intensity is `stark_shift`.
"""
function tweezer_trap_potential(x::Real,y::Real, z::Real, tweezer_trap::CrossedTweezerTrap)
    stark_shift = tweezer_trap.stark_shifts.ground_state
    return stark_shift * tweezer_trap_intensity(x, y, z, tweezer_trap)
end

"""
    fiber_potential(x::Real, y::Real, z::Real, l::Integer, f::Integer, fiber::Fiber, polarization_basis::CircularPolarization, power::Real, stark_shift::Real)

Compute the potential at position (`x`, `y`, `z`) of a circularly polarized fiber mode along
the z-axis with polarization index `l`, direction of propagation index `f`, `power`, and the
stark_shift per unit intensity is `stark_shift`.
"""
function fiber_potential(x::Real, y::Real, z::Real, f::Integer, fiber::Fiber, polarization::Polarization, power::Real, stark_shift::Real)
    ρ = sqrt(x^2 + y^2)
    ϕ = atan(y, x)
    e_x, e_y, e_z = electric_guided_field_cartesian_components(ρ, ϕ, z, 0.0, f, fiber, polarization, power)
    e_abs2 = abs2(e_x) + abs2(e_y) + abs2(e_z)
    return stark_shift * e_abs2
end

"""
    full_potential(x::Real, y::Real, z::Real, waist::Real, power_trap::Real, wavelength::Real, stark_shift_trap::Real, l::Integer, f::Integer, fiber::Fiber, polarization_basis::CircularPolarization, power_fiber::Real, stark_shift_fiber::Real)

Compute the full potential at position (`x`, `y`, `z`) of a circularly polarized fiber mode
along the z-axis and a dipole tweezer trap consisting of two beam parallel to the gaussian
beams, one along the x-axis and one along the y-axis.
"""
function full_potential(x::Real, y::Real, z::Real, tweezer_trap::CrossedTweezerTrap, f::Integer, fiber::Fiber, polarization::Polarization, power_fiber::Real, stark_shift_fiber::Real)
    V_trap = tweezer_trap_potential(x, y, z, tweezer_trap)
    V_fiber = fiber_potential(x, y, z, f, fiber, polarization, power_fiber, stark_shift_fiber)
    return V_trap + V_fiber
end

function atomic_propability_distribution(x::Real, y::Real, z::Real, tweezer_trap::CrossedTweezerTrap, f::Integer, fiber::Fiber, polarization::Polarization, power_fiber::Real, stark_shift_fiber::Real, temperature::Real)
    ρ = sqrt(x^2 + y^2)
    if ρ < radius(fiber)
        return 0.0
    end
    
    V_total = full_potential(x, y, z, tweezer_trap, f, fiber, polarization, power_fiber, stark_shift_fiber)
    if V_total > 0.0
        return 0.0
    end

    P, _ = gamma_inc(3 / 2, -V_total / temperature)
    propability = P * exp(-V_total / temperature)

    return propability
end

function atomic_propability_distribution_approximation(x::Real, y::Real, z::Real, tweezer_trap::CrossedTweezerTrap, f::Integer, fiber::Fiber, polarization::Polarization, power_fiber::Real, stark_shift_fiber::Real, temperature::Real)
    ρ = sqrt(x^2 + y^2)
    if ρ < radius(fiber)
        return 0.0
    end
    
    V_total = full_potential(x, y, z, tweezer_trap, f, fiber, polarization, power_fiber, stark_shift_fiber)    
    propability = exp(-V_total / temperature)

    return propability
end

function atomic_propability_normalization_constant(tweezer_trap::CrossedTweezerTrap, f, fiber, polarization, power_fiber, stark_shift_fiber, temperature)
    L = 2 * waist
    domain = ([-L, -L, -L], [L, L, L])
    p = tweezer_trap, f, fiber, polarization, power_fiber, stark_shift_fiber, temperature
    prob = IntegralProblem(atomic_propability_normalization_constant_integrand, domain, p)
    sol = solve(prob, CubaDivonne())
    return sol.u
end

function atomic_propability_normalization_constant_approximation(tweezer_trap::CrossedTweezerTrap, f, fiber, polarization, power_fiber, stark_shift_fiber, temperature)
    L = 2 * waist
    domain = ([-L, -L, -L], [L, L, L])
    p = tweezer_trap, f, fiber, polarization, power_fiber, stark_shift_fiber, temperature
    prob = IntegralProblem(atomic_propability_normalization_constant_approximation_integrand, domain, p)
    sol = solve(prob, CubaDivonne())
    return sol.u
end

function atomic_propability_normalization_constant_integrand(u, p)
    tweezer_trap, f, fiber, polarization, power_fiber, stark_shift_fiber, temperature = p
    return atomic_propability_distribution(u[1], u[2], u[3], tweezer_trap, f, fiber, polarization, power_fiber, stark_shift_fiber, temperature)
end

function atomic_propability_normalization_constant_approximation_integrand(u, p)
    tweezer_trap, f, fiber, polarization, power_fiber, stark_shift_fiber, temperature = p
    return atomic_propability_distribution_approximation(u[1], u[2], u[3], tweezer_trap, f, fiber, polarization, power_fiber, stark_shift_fiber, temperature)
end

function maximum_within_box(f, r0, L, p; algorithm=LBFGS())
    lower = fill(-L, 3)
    upper = fill(L, 3)
    opt = optimize(r -> -f(r, p), lower, upper, r0, Fminbox(algorithm))
    x_max = Optim.minimizer(opt)
    return f(x_max, p)
end

function rejection_sample(f, L, N, p; r0=zeros(3), algorithm=LBFGS())
    fmax = maximum_within_box(f, r0, L, p, algorithm=algorithm)
    samples = zeros(3, N)
    count = 0
    while count < N
        r = (rand(3) .- 0.5) .* (2L)
        if rand() <= f(r, p) / fmax
            count += 1
            samples[:, count] = r
        end
    end
    return samples
end

function atomic_cloud_full_potential(N, tweezer_trap::CrossedTweezerTrap, ϕ₀, f::Integer, fiber::Fiber, polarization_basis::LinearPolarization, power_fiber::Real, stark_shift_fiber::Real, temperature::Real)
    p = (tweezer_trap, ϕ₀, f, fiber, polarization_basis, power_fiber, stark_shift_fiber, temperature)
    f_unnorm(r, p) = OpticalFibers.atomic_propability_distribution(r..., p...)
    
    L = 2 * waist_trap
    a = fiber.radius
    x0 = [2a, 2a, 0.0]
    return rejection_sample(f_unnorm, L, N, p, r0=x0)
end

function atomic_propability_distribution_wrap(r, p)
    tweezer_trap, ϕ₀_control, f, fiber_control, polarization_basis, P_control, ζ_control_ground, temperature = p
    return OpticalFibers.atomic_propability_distribution(r..., tweezer_trap, ϕ₀_control, f, fiber_control, polarization_basis, P_control, ζ_control_ground, temperature)
end

function atom_number(peak_density_μm, tweezer_trap::CrossedTweezerTrap, f, fiber_control, polarization, P_control, ζ_control_ground, temperature)
    a = fiber_control.radius
    p = w_trap, P_trap, λ_trap, ζ_trap_ground, f, fiber_control, polarization, P_control, ζ_control_ground, temperature
    normalization_constant = OpticalFibers.atomic_propability_normalization_constant(tweezer_trap, f, fiber_control, polarization, P_control, ζ_control_ground, temperature)
    P_M = OpticalFibers.maximum_within_box(atomic_propability_distribution_wrap, [2a, 2a, 2a], 2 * w_trap, p; algorithm=LBFGS()) / normalization_constant
    return round(Int, peak_density_μm / P_M)
end
