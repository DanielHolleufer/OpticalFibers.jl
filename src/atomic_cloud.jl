function box_maximization(f, r0, lower, upper, p; algorithm=LBFGS())
    opt = Optim.optimize(r -> -f(r, p), lower, upper, r0, Fminbox(algorithm))
    x_max = Optim.minimizer(opt)
    return f(x_max, p)
end

function box_rejection_sampling(f, lower, upper, N, p; r0=zeros(3), algorithm=LBFGS())
    fmax = box_maximization(f, r0, lower, upper, p, algorithm=algorithm)
    samples = Matrix{Float64}(undef, 3, N)
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

abstract type AtomicCloud end

function atomic_density_distribution(r, cloud::AtomicCloud)
    return atomic_density_distribution(r..., cloud)
end

function gaussian_cloud_unnormalized(x, y, z, p)
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
    prob = IntegralProblem((u, p) -> gaussian_cloud_unnormalized(u..., p), domain, p)
    sol = solve(prob, CubaDivonne())
    return inv(sol.u)
end

struct GaussianCloud <: AtomicCloud
    σ_x::Float64
    σ_y::Float64
    σ_z::Float64
    fiber_radius::Float64
    exclusion_zone::Float64
    peak_density::Float64
    number_of_atoms::Int
    normalization_constant::Float64
    function GaussianCloud(
        σ_x::Float64,
        σ_y::Float64,
        σ_z::Float64,
        fiber_radius::Float64,
        exclusion_zone::Float64,
        peak_density::Float64,
    )
        σ_x ≤ 0 && throw(DomainError(σ_x, "Standard deviation σ_x must be positive."))
        σ_y ≤ 0 && throw(DomainError(σ_y, "Standard deviation σ_y must be positive."))
        σ_z ≤ 0 && throw(DomainError(σ_z, "Standard deviation σ_z must be positive."))
        peak_density ≤ 0 && throw(DomainError(peak_density, "Peak density must be positive."))
        fiber_radius ≤ 0 && throw(DomainError(fiber_radius, "Fiber radius must be positive."))
        exclusion_zone < 0 && throw(DomainError(exclusion_zone, "Exclusion zone must be non-negative."))

        p = (σ_x, σ_y, σ_z, fiber_radius, exclusion_zone)
        normalization_constant = gaussian_cloud_normalization_constant(p)        
        a = fiber_radius + exclusion_zone + eps(fiber_radius + exclusion_zone)
        peak_probability = normalization_constant * exp(-a^2 / (2 * max(σ_x, σ_y)^2))
        number_of_atoms = round(Int, peak_density / (peak_probability))

        return new(
            σ_x,
            σ_y,
            σ_z,
            fiber_radius,
            exclusion_zone,
            peak_density,
            number_of_atoms,
            normalization_constant,
        )
    end
end

function GaussianCloud(
    σ_x::Real,
    σ_y::Real,
    σ_z::Real,
    fiber_radius::Real,
    exclusion_zone::Real,
    peak_density::Real,
)
    return GaussianCloud(
        Float64(σ_x),
        Float64(σ_y),
        Float64(σ_z),
        Float64(fiber_radius),
        Float64(exclusion_zone),
        Float64(peak_density),
    )
end

function GaussianCloud(
    σ_x::Real, σ_y::Real, σ_z::Real, fiber::Fiber, exclusion_zone::Real, peak_density::Real
)
    return GaussianCloud(σ_x, σ_y, σ_z, fiber.radius, exclusion_zone, peak_density)
end

peak_density(cloud::GaussianCloud) = cloud.peak_density
number_of_atoms(cloud::GaussianCloud) = cloud.number_of_atoms

"""
    peak_density_gaussian_cloud(σ_x::Real, σ_y::Real, σ_z::Real, fiber_radius::Real,
                                exclusion_zone::Real, number_of_atoms::Int)
        
Compute the peak density of a Gaussian cloud with the given parameters.
"""
function peak_density_gaussian_cloud(
    σ_x::Real,
    σ_y::Real,
    σ_z::Real,
    fiber_radius::Real,
    exclusion_zone::Real,
    number_of_atoms::Int
)
    p = (σ_x, σ_y, σ_z, fiber_radius, exclusion_zone)
    normalization_constant = gaussian_cloud_normalization_constant(p)
    lower = [-5 * σ_x, -5 * σ_y, -5 * σ_z]
    upper = [5 * σ_x, 5 * σ_y, 5 * σ_z]
    P_M = box_maximization(
        (u, p) -> gaussian_cloud_unnormalized(u..., p),
        [fiber_radius + exclusion_zone, 0.0, 0.0],
        lower,
        upper,
        p,
    )
    peak_density = normalization_constant * P_M * number_of_atoms
    return peak_density
end

function peak_density_gaussian_cloud(
    σ_x::Real,
    σ_y::Real,
    σ_z::Real,
    fiber::Fiber,
    exclusion_zone::Real, 
    number_of_atoms::Int,
)
    return peak_density_gaussian_cloud(
        σ_x, σ_y, σ_z, fiber.radius, exclusion_zone, number_of_atoms,
    )
end

function Base.show(io::IO, cloud::GaussianCloud)
    println(io, "Gaussian cloud parameters:")
    println(io, "σ_x = $(cloud.σ_x)")
    println(io, "σ_y = $(cloud.σ_y)")
    println(io, "σ_z = $(cloud.σ_z)")
    println(io, "Peak density = $(cloud.peak_density)")
    println(io, "Fiber radius = $(cloud.fiber_radius)")
    println(io, "Exclusion zone = $(cloud.exclusion_zone)")
    print(io, "Number of atoms = $(cloud.number_of_atoms)")
end

function atomic_density_distribution(x, y, z, cloud::GaussianCloud)
    σ_x, σ_y, σ_z = cloud.σ_x, cloud.σ_y, cloud.σ_z
    fiber_radius = cloud.fiber_radius
    exclusion_zone = cloud.exclusion_zone
    p = (σ_x, σ_y, σ_z, fiber_radius, exclusion_zone)
    n = cloud.normalization_constant
    return n * gaussian_cloud_unnormalized(x, y, z, p)
end

function atomic_cloud(cloud::GaussianCloud)
    N = cloud.number_of_atoms
    σ_x, σ_y, σ_z = cloud.σ_x, cloud.σ_y, cloud.σ_z
    fiber_radius = cloud.fiber_radius
    exclusion_zone = cloud.exclusion_zone
    lower = [-5 * σ_x, -5 * σ_y, -5 * σ_z]
    upper = [5 * σ_x, 5 * σ_y, 5 * σ_z]
    
    samples = box_rejection_sampling(
        atomic_density_distribution, lower, upper, N, cloud;
        r0=[fiber_radius + exclusion_zone, 0.0, 0.0]
    )
    
    return samples
end

function crossed_tweezer_fiber_cloud_unnormalized(x::Real, y::Real, z::Real, p)
    crossed_tweezer_fiber_trap, temperature, exclusion_zone = p

    ρ = sqrt(x^2 + y^2)
    if ρ < crossed_tweezer_fiber_trap.fiber_trap.fiber.radius + exclusion_zone
        return 0.0
    end
    
    V_total = trap_potential(x, y, z, crossed_tweezer_fiber_trap)
    if V_total > 0.0
        return 0.0
    end

    P, _ = gamma_inc(3 / 2, -V_total / temperature)
    propability = P * exp(-V_total / temperature)

    return propability
end

function crossed_tweezer_fiber_cloud_normalization(p)
    trap = p[1]
    L = 2 * trap.tweezer_trap.waist
    domain = ([-L, -L, -L], [L, L, L])
    prob = IntegralProblem(
        (u, p) -> crossed_tweezer_fiber_cloud_unnormalized(u..., p), domain, cloud
    )
    sol = solve(prob, CubaDivonne())
    return inv(sol.u)
end

struct CrossedTweezerFiberTrappedCloud <: AtomicCloud
    trap::CrossedTweezerFiberTrap
    temperature::Float64
    exclusion_zone::Float64
    peak_density::Float64
    number_of_atoms::Int
    normalization_constant::Float64
    function CrossedTweezerFiberTrappedCloud(
        trap::CrossedTweezerFiberTrap,
        temperature::Float64,
        exclusion_zone::Float64,
        peak_density::Float64,
    )
        peak_density < 0 && throw(DomainError(peak_density, "Peak density must be \
                                                             non-negative."))
        temperature < 0 && throw(DomainError(σ_x, "Standard deviation σ_x must be \
                                                   non-negative."))
        exclusion_zone < 0 && throw(DomainError(exclusion_zone, "Exclusion zone must be \
                                                                 non-negative."))

        p = (trap, temperature, exclusion_zone)
        normalization_constant = crossed_tweezer_fiber_cloud_normalization(p)
        L = 2 * trap.tweezer_trap.waist
        lower = [-2 * L, -2 * L, -2 * L]
        upper = [2 * L, 2 * L, 2 * L]
        P_M = box_maximization(
            (u, p) -> crossed_tweezer_fiber_cloud_unnormalized(u..., p),
            [trap.tweezer_trap.waist, 0.0, 0.0],
            lower,
            upper,
            p,
        )
        number_of_atoms = round(Int, peak_density / (normalization_constant * P_M))

        return new(
            trap,
            temperature,
            exclusion_zone,
            peak_density,
            number_of_atoms,
            normalization_constant,
        )
    end
end

function atomic_density_distribution(
    x::Real, y::Real, z::Real, cloud::CrossedTweezerFiberTrappedCloud
)
    p = (cloud.trap, cloud.temperature, cloud.exclusion_zone)
    n = cloud.normalization_constant
    return n * gaussian_cloud_unnormalized(x, y, z, p)
end

function crossed_tweezer_fiber_cloud_approx_unnormalized(
    x::Real, y::Real, z::Real, cloud::CrossedTweezerFiberTrappedCloud
)
    ρ = sqrt(x^2 + y^2)
    if ρ < radius(fiber)
        return 0.0
    end
    
    V_total = ftrap_potential(x, y, z, cloud.trap)  
    propability = exp(-V_total / temperature)

    return propability
end

function atomic_density_normalization_constant_approximation(
    cloud::CrossedTweezerFiberTrappedCloud,
)
    L = 2 * cloud.trap.tweezer_trap.waist
    domain = ([-L, -L, -L], [L, L, L])
    prob = IntegralProblem(
        atomic_density_distribution_approximation_unnormalized, domain, cloud
    )
    sol = solve(prob, CubaDivonne())
    return sol.u
end

struct LinearChain <: AtomicCloud
    radial_distance::Float64
    angle::Float64
    z₀::Float64
    σ_x::Float64
    σ_y::Float64
    σ_z::Float64
    lattice_constant::Float64
    fiber_radius::Float64
    number_of_atoms::Int
    number_of_sites::Int
end

function Base.show(io::IO, cloud::LinearChain)
    println(io, "Linear chain parameters:")
    println(io, "ρ = $(cloud.radial_distance)")
    println(io, "ϕ = $(cloud.angle)")
    println(io, "z₀ = $(cloud.z₀)")
    println(io, "σ_x = $(cloud.σ_x)")
    println(io, "σ_y = $(cloud.σ_y)")
    println(io, "σ_z = $(cloud.σ_z)")
    println(io, "Lattice constant = $(cloud.lattice_constant)")
    println(io, "Fiber radius = $(cloud.fiber_radius)")
    println(io, "Number of atoms = $(cloud.number_of_atoms)")
    print(io, "Number of sites = $(cloud.number_of_sites)")
end

function atomic_cloud(cloud::LinearChain)
    ρ = cloud.radial_distance
    ϕ = cloud.angle
    z₀ = cloud.z₀
    d = cloud.lattice_constant
    Na = cloud.number_of_atoms
    Ns = cloud.number_of_sites

    normal_x = Normal(0.0, cloud.σ_x)
    normal_y = Normal(0.0, cloud.σ_y)
    normal_z = Normal(0.0, cloud.σ_z)
    
    x = ρ * cos(ϕ)
    y = ρ * sin(ϕ)

    positions  = Matrix{Float64}(undef, 3, Ns)
    for i in 1:Ns
        while true
            dx = rand(normal_x)
            dy = rand(normal_y)
            (x + dx)^2 + (y + dy)^2 > cloud.fiber_radius^2 && break
        end

        positions[1, i] = x + rand(normal_x)
        positions[2, i] = y + rand(normal_y)
        positions[3, i] = z₀ + (i - 1) * d + rand(normal_z)
    end

    filled_sites = sort(randperm(Ns)[1:Na])
    positions = positions[:, filled_sites]

    return positions
end
