function box_maximization(f, r0, lower, upper, p; algorithm=LBFGS())
    opt = optimize(r -> -f(r, p), lower, upper, r0, Fminbox(algorithm))
    x_max = Optim.minimizer(opt)
    return f(x_max, p)
end

function box_rejection_sampling(f, lower, upper, N, p; r0=zeros(3), algorithm=LBFGS())
    fmax = box_maximization(f, r0, lower, upper, p, algorithm=algorithm)
    samples = zeros(3, N)
    count = 0
    while count < N
        r = rand(3) .* (upper - lower) + lower
        if rand() <= f(r, p) / fmax
            count += 1
            samples[:, count] = r
        end
    end
    return samples
end

struct GaussianCloud
    σ_x::Float64
    σ_y::Float64
    σ_z::Float64
    peak_density::Float64
    fiber_radius::Float64
    exclusion_zone::Float64
    atoms::Int
    normalization_constant::Float64
    function GaussianCloud(σ_x::Float64, σ_y::Float64, σ_z::Float64, peak_density::Float64, 
                           fiber_radius::Float64, exclusion_zone::Float64)
        σ_x < 0 && throw(DomainError(σ_x, "Standard deviation σ_x must be non-negative."))
        σ_y < 0 && throw(DomainError(σ_y, "Standard deviation σ_y must be non-negative."))
        σ_z < 0 && throw(DomainError(σ_z, "Standard deviation σ_z must be non-negative."))
        peak_density < 0 && throw(DomainError(peak_density, "Peak density must be non-negative."))
        fiber_radius < 0 && throw(DomainError(fiber_radius, "Fiber radius must be non-negative."))
        exclusion_zone < 0 && throw(DomainError(exclusion_zone, "Exclusion zone must be non-negative."))

        p = σ_x, σ_y, σ_z, fiber_radius, exclusion_zone
        normalization_constant = gaussian_cloud_normalization_constant(p)
        lower = [-5 * σ_x, -5 * σ_y, -5 * σ_z]
        upper = [5 * σ_x, 5 * σ_y, 5 * σ_z]
        P_M = box_maximization(gaussian_cloud_distribution_unnormalized,
                               [fiber_radius, 0.0, 0.0], lower, upper, p; algorithm=LBFGS())
        atoms = round(Int, normalization_constant * peak_density / P_M)

        return new(σ_x, σ_y, σ_z, peak_density, fiber_radius, exclusion_zone, atoms,
                   normalization_constant)
    end
end

function GaussianCloud(σ_x::Real, σ_y::Real, σ_z::Real, peak_density::Real,
                       fiber_radius::Real, exclusion_zone::Real = 0.0)
    return GaussianCloud(Float64(σ_x), Float64(σ_y), Float64(σ_z), Float64(peak_density),
                         Float64(fiber_radius), Float64(exclusion_zone))
end

function GaussianCloud(σ_x::Real, σ_y::Real, σ_z::Real, peak_density::Real, fiber::Fiber,
                       exclusion_zone::Real = 0.0)
    return GaussianCloud(σ_x, σ_y, σ_z, peak_density, fiber.radius, exclusion_zone)
end

function Base.show(io::IO, cloud::GaussianCloud)
    println(io, "Gaussian cloud parameters:")
    println(io, "σ_x = $(cloud.σ_x)")
    println(io, "σ_y = $(cloud.σ_y)")
    println(io, "σ_z = $(cloud.σ_z)")
    println(io, "Peak density = $(cloud.peak_density)")
    println(io, "Fiber radius = $(cloud.fiber_radius)")
    println(io, "Exclusion zone = $(cloud.exclusion_zone)")
    print(io, "Number of atoms = $(cloud.atoms)")
end

function gaussian_cloud_distribution_unnormalized(u, p)
    x, y, z = u
    σ_x, σ_y, σ_z, fiber_radius, exclusion_zone = p
    if sqrt(x^2 + y^2) ≤ fiber_radius + exclusion_zone
        return 0.0
    else
        return exp(-x^2 / (2 * σ_x^2) - y^2 / (2 * σ_y^2) - z^2 / (2 * σ_z^2))
    end
end

function gaussian_cloud_normalization_constant(p)
    σ_x, σ_y, σ_z, _, _ = p
    domain = ([-5 * σ_x, -5 * σ_y, -5 * σ_z], [5 * σ_x, 5 * σ_y, 5 * σ_z])
    prob = IntegralProblem(gaussian_cloud_distribution_unnormalized, domain, p)
    sol = solve(prob, CubaDivonne())
    return sol.u
end

function atomic_density_distribution(r, cloud::GaussianCloud)
    x, y, z = r
    σ_x, σ_y, σ_z = cloud.σ_x, cloud.σ_y, cloud.σ_z
    fiber_radius = cloud.fiber_radius
    exclusion_zone = cloud.exclusion_zone
    n = cloud.normalization_constant
    if sqrt(x^2 + y^2) ≤ fiber_radius + exclusion_zone
        return 0.0
    else
        return n * exp(-x^2 / (2 * σ_x^2) - y^2 / (2 * σ_y^2) - z^2 / (2 * σ_z^2))
    end
end

function gaussian_atomic_cloud(cloud::GaussianCloud)
    N = cloud.atoms
    σ_x, σ_y, σ_z = cloud.σ_x, cloud.σ_y, cloud.σ_z
    lower = [-5 * σ_x, -5 * σ_y, -5 * σ_z]
    upper = [5 * σ_x, 5 * σ_y, 5 * σ_z]
    
    samples = box_rejection_sampling(atomic_density_distribution, lower, upper, N, cloud;
                                     r0=zeros(3), algorithm=LBFGS())
    
    return samples
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
