function _loss_function_optim(u, f, xdata, ydata)
    return sum(abs2, ydata .- f.(xdata, u...))
end

function two_level_transmission_coefficient(Δ::Real, Δ_shift::Real, Γ_1D::Real, Γ_loss::Real)
    return 1 - im * Γ_1D / (Δ - Δ_shift + im * (Γ_1D + Γ_loss) / 2)
end

function three_level_transmission_coefficient(Δ::Real, Γ_1D::Real, Γ_loss::Real, Ω::Real, Δr::Real, γ::Real)
    return 1 - im * Γ_1D / (Δ + im * (Γ_1D + Γ_loss) / 2 - Ω^2 / (Δ + Δr + im * γ / 2))
end

function single_two_level_transmission(Δ::Real, Γ_1D::Real, Γ_loss::Real)
    return 1 - 4 * Γ_1D * Γ_loss / (4 * Δ^2 + (Γ_1D + Γ_loss)^2)
end

function single_two_level_transmission_fit(Δs, transmission_data, u0)
    res = optimize(u -> _loss_function_optim(u, single_two_level_transmission, Δs, transmission_data), [0.0, 0.0], [Inf, Inf], u0)
    return Optim.minimizer(res)
end

function single_three_level_transmission(Δ::Real, Γ_1D::Real, Γ_loss::Real, Ω::Real, Δr::Real, γ::Real)
    return abs2(1 - im * Γ_1D / (Δ + im * (Γ_1D + Γ_loss) / 2 - Ω^2 / (Δ + Δr + im * γ / 2)))
end

function single_three_level_transmission_fit(Δs, transmission_data, u0)
    lower_bounds = [0.0, 0.0, 0.0, -Inf, 0.0]
    upper_bounds = [Inf, Inf, Inf, Inf, Inf]
    res = optimize(u -> _loss_function_optim(u, single_three_level_transmission, Δs, transmission_data), lower_bounds, upper_bounds, u0)
    return Optim.minimizer(res)
end

function complex_fit_two_level(Δ, t, u0)
    res = optimize(u -> _loss_function_optim(u, two_level_transmission_coefficient, Δ, t), [-Inf, 0.0, 0.0], [Inf, Inf, Inf], u0)
    return Optim.minimizer(res)
end

function optical_depth(T::Real)
    T < 0 && throw(DomainError(T, "The transmission must be greater than or equal to zero."))
    T > 1 && throw(DomainError(T, "The transmission must be smaller than or equal to one."))
    return -log(T)
end

function coupling_strengths(d, positions, mode::GuidedMode)
    N = size(positions)[2]
    gs = Vector{ComplexF64}(undef, N)
    for i in 1:N
        x = positions[1, i]
        y = positions[2, i]
        z = positions[3, i]
        ρ = sqrt(x^2 + y^2)
        ϕ = atan(y, x)
        ex, ey, ez = electric_guided_mode_profile_cartesian_components(ρ, ϕ, mode)
        d_dot_e = conj(d[1]) * ex + conj(d[2]) * ey + conj(d[3]) * ez
        gs[i] = d_dot_e * exp(im * f * fiber.propagation_constant * z)
    end
    return gs
end

function rabi_frequency(ρ, ϕ, z, d, field::GuidedField)
    ex, ey, ez = electric_guided_field_cartesian_components(ρ, ϕ, z, field)
    rabi = conj(d[1]) * ex + conj(d[2]) * ey + conj(d[3]) * ez
    return rabi
end

function rabi_frequencies(d, positions, field::GuidedField)
    N = size(positions)[2]
    rabis = Vector{ComplexF64}(undef, N)
    for i in 1:N
        x = positions[1, i]
        y = positions[2, i]
        z = positions[3, i]
        ρ = sqrt(x^2 + y^2)
        ϕ = atan(y, x)
        ex, ey, ez = electric_guided_field_cartesian_components(ρ, ϕ, z, field)
        rabis[i] = conj(d[1]) * ex + conj(d[2]) * ey + conj(d[3]) * ez
    end
    return rabis
end

"""
    transmission_three_level(Δes, fiber, Δr, Ω::Number, gs, J, Γ, γ)

Compute the transmission of a cloud of three level atoms surrounding an optical fiber for 
each value of the lower transition detuning given by `Δes`, where each atom experience the
same Rabi frequency.

The parameters of the fiber are given by `fiber`, while the atoms have upper transition
detuning `Δr`, control Rabi frequenciy `Ω`, pump coupling constants `gs`, dipole-dipole
interaction matrix `J`, cross decay rate matrix `Γ`, and Rydberg to intermediate state decay
rate `γ`.
"""
function transmission_three_level(Δes, fiber, Δr, Ωs::Number, gs, J, Γ, γ)
    ω₀ = fiber.frequency
    dβ₀ = fiber.propagation_constant_derivative
    t = zeros(ComplexF64, length(Δes))
    M = hessenberg(-(J + im * Γ / 2))
    fill_transmissions_three_level!(t, M, Δes, Δr, Ωs, gs, ω₀, dβ₀, γ)
    return t
end

function fill_transmissions_three_level!(t, M, Δes, Δr, Ωs::Number, gs, ω₀, dβ₀, γ)
    for i in eachindex(t)
        t[i] = 1.0 + im * ω₀ * dβ₀ / 2 * gs' * ((M + (-Δes[i] + abs2(Ωs) / (Δes[i] + Δr + im * γ / 2)) * I) \ gs)
    end
end

"""
    transmission_three_level(Δes, fiber, Δr, Ωs::AbstractArray, gs, J, Γ, γ)

Compute the transmission of a cloud of three level atoms surrounding an optical fiber for 
each value of the lower transition detuning given by `Δes`, where the Rabi frequency can be
different from atom to atom.

The parameters of the fiber are given by `fiber`, while the atoms have upper transition
detuning `Δr`, control Rabi frequencies `Ωs`, pump coupling constants `gs`, dipole-dipole
interaction matrix `J`, cross decay rate matrix `Γ`, and Rydberg to intermediate state decay
rate `γ`.
"""
function transmission_three_level(Δes, fiber, Δr, Ωs::AbstractArray, gs, J, Γ, γ)
    ω₀ = fiber.frequency
    dβ₀ = fiber.propagation_constant_derivative
    t = zeros(ComplexF64, length(Δes))
    M = -(J + im * Γ / 2)
    fill_transmissions_three_level!(t, M, Δes, Δr, Ωs, gs, ω₀, dβ₀, Γ, γ)
    return t
end

function fill_transmissions_three_level!(t, M, Δes, Δr, Ωs::AbstractArray, gs, ω₀, dβ₀, Γ, γ)
    for i in eachindex(t)
        for j in eachindex(Ωs)
            M[j, j] = -Δes[i] - im * Γ[j, j] / 2 + abs2(Ωs[j]) / (Δes[i] + Δr + im * γ / 2)
        end

        t[i] = 1.0 + im * ω₀ * dβ₀ / 2 * gs' * (M \ gs)
    end
end

"""
    transmission_two_level(Δes, fiber, Δr, Ωs::AbstractArray, gs, J, Γ, γ)

Compute the transmission of a cloud of two level atoms surrounding an optical fiber for 
each value of the detuning given by `Δes`.

The parameters of the fiber are given by `fiber`, while the atoms have light-matter coupling
constants `gs`, dipole-dipole interaction matrix `J`, and cross decay rate matrix `Γ`.
"""
function transmission_two_level(Δes, fiber, gs, J, Γ)
    ω₀ = fiber.frequency
    dβ₀ = fiber.propagation_constant_derivative
    t = zeros(ComplexF64, length(Δes))
    M = hessenberg(-(J + im * Γ / 2))
    fill_transmissions_two_level!(t, M, Δes, gs, ω₀, dβ₀)
    return t
end

function fill_transmissions_two_level!(t, M, Δes, gs, ω₀, dβ₀)
    for i in eachindex(t)
        t[i] = 1.0 + im * ω₀ * dβ₀ / 2 * gs' * ((M - Δes[i] * I) \ gs)
    end
end

function full_width_half_minimum_index(xs, ys)
    length(xs) == length(ys) || throw(DimensionMismatch("The x and y arrays must have the same length."))

    y_min, y_min_index = findmin(ys)
    y_max = maximum(ys)

    half_min = (y_max + y_min) / 2
    half_min_lower_index = argmin(abs.(ys[1:y_min_index] .- half_min))
    half_min_upper_index = argmin(abs.(ys[y_min_index:end] .- half_min)) + y_min_index - 1

    return half_min_lower_index, half_min_upper_index
end

function full_width_half_minimum(xs, ys)
    lower_index, upper_index = full_width_half_minimum_index(xs, ys)
    return xs[upper_index] - xs[lower_index]
end

# This function is to access correct data generated before a bug, which sometimes caused an
# incorrect range to be generated, was fixed. In other words: backwards compatibility.
# It is not exported and will probably be removed in the future, once new data has been
# generated.
function probe_detuning_range_legacy(Δ_control, d_probe, d_control, Γ_probe, Γ_control,
                                     polarization_probe, polarization_control,
                                     fiber_probe::Fiber, fiber_control::Fiber, P_control,
                                     z_span, cloud::GaussianCloud, resolution::Int)
    _transmission(Δ_probe) = abs2(continuous_propagation(Δ_probe, Δ_control, d_probe,
                                                         d_control, Γ_probe, Γ_control,
                                                         polarization_probe,
                                                         polarization_control, fiber_probe,
                                                         fiber_control, P_control, z_span,
                                                         cloud))

    result_minimum = optimize(_transmission, -Δ_control - Γ_probe, -Δ_control)
    Δ_min = result_minimum.minimizer

    dark_state_transmission = _transmission(-Δ_control)
    half_minimum = (result_minimum.minimum + dark_state_transmission) / 2

    result_fwhm_left = optimize(Δ_probe -> abs(_transmission(Δ_probe) - half_minimum),
                                -Δ_control - Γ_probe, Δ_min)
    Δ_fwhm_left = result_fwhm_left.minimizer

    left_extension_factor = 2.5
    right_extension_factor = 2.0
    
    lower_bound = Δ_min - left_extension_factor * (Δ_min - Δ_fwhm_left)
    upper_bound = -Δ_control + right_extension_factor * (-Δ_control - Δ_min)

    return LinRange(lower_bound, upper_bound, resolution)
end

"""
    probe_detuning_range(Δ_control, d_probe, d_control, Γ_probe, Γ_control,
                  polarization_probe, polarization_control, fiber_probe, fiber_control,
                  P_control, z_span, cloud::GaussianCloud, resolution)

Compute a suitable range of probe detunings around the Autler Townes resonance that appear
near the two-photon resonance at low control power.
"""
function probe_detuning_range_deprecated(Δ_control, d_probe, d_control, Γ_probe, Γ_control,
                              polarization_probe, polarization_control, fiber_probe::Fiber,
                              fiber_control::Fiber, P_control, z_span, cloud::GaussianCloud,
                              resolution::Int)
    _transmission(Δ_probe) = abs2(continuous_propagation_deprecated(Δ_probe, Δ_control, d_probe,
                                                         d_control, Γ_probe, Γ_control,
                                                         polarization_probe,
                                                         polarization_control, fiber_probe,
                                                         fiber_control, P_control, z_span,
                                                         cloud))

    minimum_upper_bound = -Δ_control # Two-photon resonance
    minimum_lower_bound = -Δ_control - Γ_probe / 5 # A bit below Two-photon resonance. This
                                                   # is a robust guess that seems to work in
                                                   # all tested cases.
    result_minimum = optimize(_transmission, minimum_lower_bound, minimum_upper_bound)
    Δ_min = result_minimum.minimizer

    dark_state_transmission = _transmission(-Δ_control)
    half_minimum = (result_minimum.minimum + dark_state_transmission) / 2

    fwhm_upper_bound = Δ_min # Transmission minimum.
    fwhm_lower_bound = -Δ_control - Γ_probe / 10 # A bit below the minimum. Again, this is a
                                                 # robust guess that seems to work in all
                                                 # tested cases.
    result_fwhm_left = optimize(Δ_probe -> abs(_transmission(Δ_probe) - half_minimum),
                                fwhm_lower_bound, fwhm_upper_bound)
    Δ_fwhm_left = result_fwhm_left.minimizer

    left_extension_factor = 2.5
    right_extension_factor = 2.0
    
    lower_bound = Δ_min - left_extension_factor * (Δ_min - Δ_fwhm_left)
    upper_bound = -Δ_control + right_extension_factor * (-Δ_control - Δ_min)

    return LinRange(lower_bound, upper_bound, resolution)
end

function probe_detuning_range_deprecated(Δ_control, d_probe, Γ_probe, Γ_control,
                              polarization_probe, fiber_probe::Fiber,
                              rabi_frequency, z_span, cloud::GaussianCloud,
                              resolution::Int)
    _transmission(Δ_probe) = abs2(continuous_propagation_deprecated(Δ_probe, Δ_control, d_probe,
                                                         Γ_probe, Γ_control,
                                                         polarization_probe, fiber_probe, 
                                                         rabi_frequency, z_span, cloud))
    result_minimum = optimize(_transmission, -Δ_control - rabi_frequency, -Δ_control)
    Δ_min = result_minimum.minimizer

    half_minimum = (result_minimum.minimum + 1.0) / 2
    result_fwhm_left = optimize(Δ_probe -> abs(_transmission(Δ_probe) - half_minimum),
                                Δ_min - 2 * rabi_frequency, Δ_min)
    Δ_fwhm_left = result_fwhm_left.minimizer

    left_extension_factor = 3.0
    right_extension_factor = 3.0
    
    lower_bound = Δ_min - left_extension_factor * (Δ_min - Δ_fwhm_left)
    upper_bound = Δ_min + right_extension_factor * (Δ_min - Δ_fwhm_left)

    return LinRange(lower_bound, upper_bound, resolution)
end
