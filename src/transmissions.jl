function _loss_function_optim(u, f, xdata, ydata)
    return sum(abs2, ydata .- f.(xdata, u...))
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
    res = optimize(u -> _loss_function_optim(u, single_three_level_transmission, Δs, transmission_data), [0.0, 0.0, 0.0, -Inf, 0.0], [Inf, Inf, Inf, Inf, Inf], u0)
    return Optim.minimizer(res)
end

function optical_depth(T::Real)
    T < 0 && throw(DomainError(T, "The transmission must be greater than or equal to zero."))
    T > 1 && throw(ArgumentError("The transmission must be smaller than or equal to one."))
    return -log(T)
end

function coupling_strengths(d, r, l, f, fiber, polarization_basis::CircularPolarization)
    N = size(r)[2]
    Ωs = zeros(ComplexF64, N)
    for i in 1:N
        ρ_i = sqrt(r[1, i]^2 + r[2, i]^2)
        ϕ_i = atan(r[2, i], r[1, i])
        z_i = r[3, i]
        e_x, e_y, e_z = electric_guided_mode_cartesian_components(ρ_i, ϕ_i, l, f, fiber, polarization_basis)
        d_dot_e = conj(d[1]) * e_x + conj(d[2]) * e_y + conj(d[3]) * e_z
        Ωs[i] = d_dot_e * exp(im * l * ϕ_i) * exp(im * f * fiber.propagation_constant * z_i)
    end
    return Ωs
end

function fill_transmissions_three_level!(t, M, Δes, Δr, Ωs, gs, ω₀, dβ₀, Γ, γ)
    for i in eachindex(t)
        for j in eachindex(Ωs)
            M[j, j] = -Δes[i] - im * Γ[j, j] / 2 + abs2(Ωs[j]) / (Δes[i] + Δr + im * γ / 2)
        end

        t[i] = 1.0 + im * ω₀ * dβ₀ / 2 * gs' * (M \ gs)
    end
end

"""
    transmission_three_level(Δes, fiber, Δr, Ωs, gs, J, Γ, γ)

Compute the transmission of a cloud of three level atoms surrounding an optical fiber for 
each value of the lower transition detuning given by `Δes`.

The parameters of the fiber are given by `fiber`, while the atoms have upper transition
detuning `Δr`, control Rabi frequencies `Ωs`, pump coupling constants `gs`, dipole-dipole
interaction matrix `J`, cross decay rate matrix `Γ`, and Rydberg to intermediate state decay
rate `γ`.
"""
function transmission_three_level(Δes, fiber, Δr, Ωs, gs, J, Γ, γ)
    ω₀ = fiber.frequency
    dβ₀ = fiber.propagation_constant_derivative
    t = zeros(ComplexF64, length(Δes))
    M = -(J + im * Γ / 2)
    fill_transmissions_three_level!(t, M, Δes, Δr, Ωs, gs, ω₀, dβ₀, Γ, γ)
    return t
end
