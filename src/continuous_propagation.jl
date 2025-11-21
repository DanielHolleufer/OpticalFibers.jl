function transmission_coefficient_continuous_propagation(
    Δ_e::Float64,
    probe::GuidedMode,
    control::ElectricField,
    atom::ThreeLevelAtom,
    cloud::GaussianCloud,
)
    λ = wavelength(probe)
    ω = frequency(probe)
    db = propagation_constant_derivative(probe)
    factor = sqrt(ω * db / (4 * π))
    p = (Δ_e, probe, control, atom, cloud, factor)
    a = radius(probe) + cloud.exclusion_zone

    domain = ([a + eps(a), 0.0], [λ + a, 2π])
    prob = IntegralProblem(continuous_propagation_integrand, domain, p)
    alg = HCubatureJL()
    cache = init(prob, alg; abstol=1e-9, reltol=1e-9)
    sol_1 = solve!(cache)

    cache.domain = ([λ + a, 0.0], [10λ + a, 2π])
    sol_2 = solve!(cache)

    cache.domain = ([10λ + a, 0.0], [100λ + a, 2π])
    sol_3 = solve!(cache)

    cache.domain = ([100λ + a, 0.0], [1000λ + a, 2π])
    sol_4 = solve!(cache)

    N = cloud.number_of_atoms

    return exp(N * (sol_1.u + sol_2.u + sol_3.u + sol_4.u))
end

function transmission_coefficient_continuous_propagation(
    Δ_es::AbstractVector{Float64},
    probe::GuidedMode,
    control::ElectricField,
    atom::ThreeLevelAtom,
    cloud::GaussianCloud,
)
    ts = Vector{ComplexF64}(undef, length(Δ_es))
    for (i, Δ) in enumerate(Δ_es)
        ts[i] = transmission_coefficient_continuous_propagation(
            Δ, probe, control, atom, cloud
        )
    end

    return ts
end

function continuous_propagation_integrand(u, p)
    ρ = u[1]
    ϕ = u[2]
    Δ_e, probe, control, atom, cloud, factor = p
    σ_r = cloud.σ_x
    n = cloud.normalization_constant * sqrt(2π) * cloud.σ_z

    distribution = ρ * n * exp(-ρ^2 / (2 * σ_r^2))
    susceptibility = _susceptibility(ρ, ϕ, Δ_e, probe, control, atom, factor)
    return susceptibility * distribution
end

function _susceptibility(
    ρ::Float64,
    ϕ::Float64,
    Δ_e::Float64,
    probe::GuidedMode,
    control::ExternalField,
    atom::ThreeLevelAtom,
    front_factor::Float64,
)
    fiber = probe.fiber
    Ω = control.rabi_frequency
    Γ_ge = atom.decay_rate_lower
    Γ_re = atom.decay_rate_upper
    d_eg = atom.dipole_moment_lower
    Δ_r = atom.detuning_upper

    Γ_guided = decay_rate_guided(ρ, ϕ, d_eg, probe, front_factor)
    Γ_guided_total = decay_rate_guided_total(ρ, ϕ, d_eg, fiber, front_factor)
    Γ_tot = Γ_guided_total + Γ_ge

    return Γ_guided / (im * Δ_e - Γ_tot / 2 + abs2(Ω) / (im * (Δ_e + Δ_r) - Γ_re / 2))
end

function _susceptibility(
    ρ::Float64,
    ϕ::Float64,
    Δ_e::Float64,
    probe::GuidedMode,
    control::GuidedField,
    atom::ThreeLevelAtom,
    front_factor::Float64,
)
    fiber = probe.fiber
    Γ_ge = atom.decay_rate_lower
    Γ_re = atom.decay_rate_upper
    d_eg = atom.dipole_moment_lower
    d_re = atom.dipole_moment_upper
    Δ_r = atom.detuning_upper

    Ω = rabi_frequency(ρ, ϕ, 0.0, d_re, control)

    Γ_guided = decay_rate_guided(ρ, ϕ, d_eg, probe, front_factor)
    Γ_guided_total = decay_rate_guided_total(ρ, ϕ, d_eg, fiber, front_factor)
    Γ_tot = Γ_guided_total + Γ_ge

    return Γ_guided / (im * Δ_e - Γ_tot / 2 + abs2(Ω) / (im * (Δ_e + Δ_r) - Γ_re / 2))
end

function decay_rate_guided(ρ, ϕ, d, mode, front_factor)
    e_x, e_y, e_z = electric_guided_mode_profile_cartesian(ρ, ϕ, mode)
    g = front_factor * (d[1] * e_x + d[2] * e_y + d[3] * e_z)
    return 2π * abs2(g)
end

function decay_rate_guided_total(ρ, ϕ, d, fiber, front_factor)
    lx = LinearPolarization(0.0)
    ly = LinearPolarization(π / 2)
    f_p = 1
    f_m = -1

    mode_xp = GuidedMode(fiber, lx, f_p)
    mode_xm = GuidedMode(fiber, lx, f_m)
    mode_yp = GuidedMode(fiber, ly, f_p)
    mode_ym = GuidedMode(fiber, ly, f_m)

    e_x1, e_y1, e_z1 = electric_guided_mode_profile_cartesian(ρ, ϕ, mode_xp)
    e_x2, e_y2, e_z2 = electric_guided_mode_profile_cartesian(ρ, ϕ, mode_xm)
    e_x3, e_y3, e_z3 = electric_guided_mode_profile_cartesian(ρ, ϕ, mode_yp)
    e_x4, e_y4, e_z4 = electric_guided_mode_profile_cartesian(ρ, ϕ, mode_ym)

    g1 = d[1] * e_x1 + d[2] * e_y1 + d[3] * e_z1
    g2 = d[1] * e_x2 + d[2] * e_y2 + d[3] * e_z2
    g3 = d[1] * e_x3 + d[2] * e_y3 + d[3] * e_z3
    g4 = d[1] * e_x4 + d[2] * e_y4 + d[3] * e_z4
    
    return 2π * front_factor^2 * (abs2(g1) + abs2(g2) + abs2(g3) + abs2(g4))
end

function _transmittance(Δ, p)
    t = transmission_coefficient_continuous_propagation(Δ, p...)
    T = abs2(t)
    return T
end

function transformed_transmittance(u, shift, p)
    Δ = shift + 1 - 1 / u
    return _transmittance(Δ, p)
end

function probe_detuning_range(
    probe::GuidedMode,
    control::ElectricField,
    atom::ThreeLevelAtom,
    cloud::GaussianCloud;
    resolution::Integer=500,
    range_scale_factor::Real=4.0,
)
    Δr = atom.detuning_upper
    p = (probe, control, atom, cloud)
    sol = Optim.optimize(u -> transformed_transmittance(u, -Δr, p), 0.9, 1.0, rel_tol=eps())
    Δ_min = -Δr + 1 - 1 / Optim.minimizer(sol)
    T_min = Optim.minimum(sol)

    dark_state_transmittance = transmission_coefficient_continuous_propagation(-Δr, p...)
    half_min = (T_min + dark_state_transmittance) / 2

    sol_fwhm_left = Optim.optimize(
        u -> abs(transformed_transmittance(u, Δ_min, p) - half_min), 0.9, 1.0, rel_tol=eps()
    )
    Δ_fwhm_left = Δ_min + 1 - 1 / Optim.minimizer(sol_fwhm_left)

    sol_fwhm_right = Optim.optimize(
        u -> abs(_transmittance(u, p) - half_min), Δ_min, -Δr, rel_tol=eps()
    )
    Δ_fwhm_right = Optim.minimizer(sol_fwhm_right)
    
    range_lower_bound = Δ_min - range_scale_factor * (Δ_min - Δ_fwhm_left)
    range_upper_bound = Δ_min + range_scale_factor * (Δ_fwhm_right - Δ_min)

    return LinRange{Float64,Int}(range_lower_bound, range_upper_bound, resolution)
end
