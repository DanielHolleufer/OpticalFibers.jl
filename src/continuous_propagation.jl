function continuous_propagation(Δ_probe, Δ_control, d_probe, d_control, Γ_probe, Γ_control,
                                polarization_probe, polarization_control,
                                fiber_probe::Fiber, fiber_control::Fiber, power, z_span,
                                cloud::GaussianCloud)
    e0 = 1.0 + 0.0im
    p_integral = (Δ_probe, Δ_control, d_probe, d_control, Γ_probe, Γ_control,
                  polarization_probe, polarization_control, fiber_probe, fiber_control, 
                  power, cloud)
    ρ_max = 5 * maximum((cloud.σ_x, cloud.σ_y))
    domain = ([0.0, 0.0], [ρ_max, 2 * π])
    prob_integral = IntegralProblem(propagation_equation_integrand, domain, p_integral)
    sol_integral = solve(prob_integral, HCubatureJL())

    ω = fiber_probe.frequency
    β = fiber_probe.propagation_constant
    α = ω^2 / (2 * β) * sol_integral.u

    p_ode = (cloud, α)
    prob_ode = ODEProblem(propagation_equation, e0, z_span, p_ode)
    sol_ode = solve(prob_ode, abstol = 1e-6, reltol = 1e-6)

    return sol_ode.u[end]
end

function propagation_equation(e, p, z)
    cloud, α = p
    return α * e * exp(-z^2 / (2 * cloud.σ_z^2))
end

function propagation_equation_integrand(u, p)
    ρ, ϕ = u
    Δ_probe, Δ_control, d_probe, d_control, Γ_probe, Γ_control, polarization_probe,
    polarization_control, fiber_probe, fiber_control, power, cloud = p
    x = ρ * cos(ϕ)
    y = ρ * sin(ϕ)
    n = cloud.number_of_atoms * atomic_density_distribution(x, y, 0.0, cloud)

    Ex_probe, Ey_probe, Ez_probe = electric_guided_mode_profile_cartesian_components(ρ, ϕ, 1, fiber_probe, polarization_probe)
    intensity_probe = abs2(Ex_probe) + abs2(Ey_probe) + abs2(Ez_probe)
    de2 = abs2(conj(d_probe[1]) * Ex_probe + conj(d_probe[2]) * Ey_probe + conj(d_probe[3]) * Ez_probe) / intensity_probe

    # The z-dependence of the field only gives a phase factor, and since we only use
    # the modulus of the field in the integrand, we can ignore it. This is why the field is
    # evaluated at z = 0.0.
    Ex_control, Ey_control, Ez_control = electric_guided_field_cartesian_components(ρ, ϕ, 0.0, 0.0, -1, fiber_control, polarization_control, power)
    Ω = conj(d_control[1]) * Ex_control + conj(d_control[2]) * Ey_control + conj(d_control[3]) * Ez_control

    integrand = ρ * n * de2 / (im * Δ_probe - Γ_probe / 2 + abs2(Ω) / (im * (Δ_probe + Δ_control) - Γ_control / 2)) * intensity_probe
    return integrand
end

function continuous_propagation(Δ_probe, Δ_control, d_probe, Γ_probe, Γ_control,
                                polarization_probe, fiber_probe::Fiber, rabi_frequency,
                                z_span, cloud::GaussianCloud)
    e0 = 1.0 + 0.0im
    p_integral = (Δ_probe, Δ_control, d_probe, Γ_probe, Γ_control, polarization_probe,
                  fiber_probe, rabi_frequency, cloud)
    ρ_max = 5 * maximum((cloud.σ_x, cloud.σ_y))
    domain = ([0.0, 0.0], [ρ_max, 2 * π])
    prob_integral = IntegralProblem(propagation_equation_integrand_constant_rabi, domain, p_integral)
    sol_integral = solve(prob_integral, HCubatureJL())

    ω = fiber_probe.frequency
    β = fiber_probe.propagation_constant
    α = ω^2 / (2 * β) * sol_integral.u

    p_ode = (cloud, α)
    prob_ode = ODEProblem(propagation_equation, e0, z_span, p_ode)
    sol_ode = solve(prob_ode, abstol = 1e-6, reltol = 1e-6)

    return sol_ode.u[end]
end

function propagation_equation_integrand_constant_rabi(u, p)
    ρ, ϕ = u
    Δ_probe, Δ_control, d_probe, Γ_probe, Γ_control, polarization_probe, fiber_probe, 
    rabi_frequency, cloud = p
    x = ρ * cos(ϕ)
    y = ρ * sin(ϕ)
    n = cloud.number_of_atoms * atomic_density_distribution(x, y, 0.0, cloud)

    Ex_probe, Ey_probe, Ez_probe = electric_guided_mode_profile_cartesian_components(ρ, ϕ, 1, fiber_probe, polarization_probe)
    intensity_probe = abs2(Ex_probe) + abs2(Ey_probe) + abs2(Ez_probe)
    de2 = abs2(conj(d_probe[1]) * Ex_probe + conj(d_probe[2]) * Ey_probe + conj(d_probe[3]) * Ez_probe) / intensity_probe

    integrand = ρ * n * de2 / (im * Δ_probe - Γ_probe / 2 + abs2(rabi_frequency) / (im * (Δ_probe + Δ_control) - Γ_control / 2)) * intensity_probe
    return integrand
end
