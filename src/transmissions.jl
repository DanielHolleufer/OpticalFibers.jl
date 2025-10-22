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
    f = direction(mode)
    β = propagation_constant(mode)
    for i in 1:N
        x = positions[1, i]
        y = positions[2, i]
        z = positions[3, i]
        ρ = sqrt(x^2 + y^2)
        ϕ = atan(y, x)
        ex, ey, ez = electric_guided_mode_profile_cartesian(ρ, ϕ, mode)
        d_dot_e = conj(d[1]) * ex + conj(d[2]) * ey + conj(d[3]) * ez
        gs[i] = d_dot_e * exp(im * f * β * z)
    end
    return gs
end

function rabi_frequency(ρ, ϕ, z, d, field::GuidedField)
    ex, ey, ez = electric_guided_field_cartesian(ρ, ϕ, z, field)
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
        ex, ey, ez = electric_guided_field_cartesian(ρ, ϕ, z, field)
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
    transmission_two_level(Δes, fiber, gs::AbstractArray, J, Γ)

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
