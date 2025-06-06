function continuous_propagation(Δ_probe, Δ_control, d_probe, d_control, Γ_probe, Γ_control,
                                polarization_probe, polarization_control, std_x, std_y,
                                std_z, fiber_probe, exclusion_zone, power,
                                normalization_constant, z_span, atom_number)
    e0 = 1.0 + 0.0im
    p = (Δ_probe, Δ_control, d_probe, d_control, Γ_probe, Γ_control, polarization_probe,
         polarization_control, std_x, std_y, std_z, fiber_probe, exclusion_zone, power,
         normalization_constant, atom_number)
    
    prob = ODEProblem(propagation_equation, e0, z_span, p)
    sol = solve(prob, abstol = 1e-4, reltol = 1e-4)
    return sol.u[end]
end

function propagation_equation(e, p, z)
    Δ_probe, Δ_control, d_probe, d_control, Γ_probe, Γ_control, polarization_probe,
    polarization_control, std_x, std_y, std_z, fiber_probe, exclusion_zone, power,
    normalization_constant, atom_number = p

    ω = fiber_probe.frequency
    β = fiber_probe.propagation_constant

    p_integral = (z, Δ_probe, Δ_control, d_probe, d_control, Γ_probe, Γ_control,
                  polarization_probe, polarization_control, std_x, std_y, std_z,
                  fiber_probe, exclusion_zone, power, normalization_constant,
                  atom_number)
    
    domain = ([0.0, 0.0], [50.0, 2 * π])
    prob = IntegralProblem(propagation_equation_integrand, domain, p_integral)
    sol = solve(prob, HCubatureJL())

    return ω^2 / (2 * β) * sol.u * e
end

function propagation_equation_integrand(u, p)
    ρ, ϕ = u
    z, Δ_probe, Δ_control, d_probe, d_control, Γ_probe, Γ_control, polarization_probe,
    polarization_control, std_x, std_y, std_z, fiber_probe, exclusion_zone, power,
    normalization_constant, atom_number = p
    x = ρ * cos(ϕ)
    y = ρ * sin(ϕ)
    n = atom_number / normalization_constant * gaussian_cloud_distribution(x, y, z, std_x, std_y, std_z, fiber_probe, exclusion_zone)

    Ex_probe, Ey_probe, Ez_probe = electric_guided_mode_profile_cartesian_components(ρ, ϕ, 1, fiber_probe, polarization_probe)
    intensity_probe = abs2(Ex_probe) + abs2(Ey_probe) + abs2(Ez_probe)
    de2 = abs2(conj(d_probe[1]) * Ex_probe + conj(d_probe[2]) * Ey_probe + conj(d_probe[3]) * Ez_probe) / intensity_probe

    Ex_control, Ey_control, Ez_control = electric_guided_field_cartesian_components(ρ, ϕ, z, 0.0, -1, fiber_probe, polarization_control, power)
    Ω = conj(d_control[1]) * Ex_control + conj(d_control[2]) * Ey_control + conj(d_control[3]) * Ez_control

    integrand = ρ * n * de2 / (im * Δ_probe - Γ_probe / 2 + abs2(Ω) / (im * (Δ_probe + Δ_control) - Γ_control / 2)) * intensity_probe
    return integrand
end
