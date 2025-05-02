function _loss_function_optim(u, f, xdata, ydata)
    return sum(abs2, ydata .- f.(xdata, u...))
end

function two_level_transmission_amplitude(Δ::Real, Δ_shift::Real, Γ_1D::Real, Γ_loss::Real)
    return 1 - im * Γ_1D / (Δ - Δ_shift + im * (Γ_1D + Γ_loss) / 2)
end

function three_level_transmission_amplitude(Δ::Real, Γ_1D::Real, Γ_loss::Real, Ω::Real, Δr::Real, γ::Real)
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
    res = optimize(u -> _loss_function_optim(u, single_three_level_transmission, Δs, transmission_data), [0.0, 0.0, 0.0, -Inf, 0.0], [Inf, Inf, Inf, Inf, Inf], u0)
    return Optim.minimizer(res)
end

function complex_fit_two_level(Δ, t, u0)
    res = optimize(u -> _loss_function_optim(u, two_level_transmission_amplitude, Δ, t), [-Inf, 0.0, 0.0], [Inf, Inf, Inf], u0)
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
        e_x, e_y, e_z = electric_guided_mode_profile_cartesian_components(ρ_i, ϕ_i, l, f, fiber, polarization_basis)
        d_dot_e = conj(d[1]) * e_x + conj(d[2]) * e_y + conj(d[3]) * e_z
        Ωs[i] = d_dot_e * exp(im * f * fiber.propagation_constant * z_i)
    end
    return Ωs
end

function coupling_strengths(d::Vector{Vector{ComplexF64}}, r, l, f, fiber, polarization_basis::CircularPolarization)
    N = size(r)[2]
    Ωs = zeros(ComplexF64, N)
    for i in 1:N
        ρ_i = sqrt(r[1, i]^2 + r[2, i]^2)
        ϕ_i = atan(r[2, i], r[1, i])
        z_i = r[3, i]
        e_x, e_y, e_z = electric_guided_mode_profile_cartesian_components(ρ_i, ϕ_i, l, f, fiber, polarization_basis)
        d_dot_e = conj(d[i][1]) * e_x + conj(d[i][2]) * e_y + conj(d[i][3]) * e_z
        Ωs[i] = d_dot_e * exp(im * f * fiber.propagation_constant * z_i)
    end
    return Ωs
end

function coupling_strengths(d, r, ϕ₀, f, fiber, polarization_basis::LinearPolarization)
    N = size(r)[2]
    Ωs = zeros(ComplexF64, N)
    for i in 1:N
        ρ_i = sqrt(r[1, i]^2 + r[2, i]^2)
        ϕ_i = atan(r[2, i], r[1, i])
        z_i = r[3, i]
        e_x, e_y, e_z = electric_guided_mode_profile_cartesian_components(ρ_i, ϕ_i, ϕ₀, f, fiber, polarization_basis)
        d_dot_e = conj(d[1]) * e_x + conj(d[2]) * e_y + conj(d[3]) * e_z
        Ωs[i] = d_dot_e * exp(im * f * fiber.propagation_constant * z_i)
    end
    return Ωs
end

function coupling_strengths(d::Vector{Vector{ComplexF64}}, r, ϕ₀, f, fiber, polarization_basis::LinearPolarization)
    N = size(r)[2]
    Ωs = zeros(ComplexF64, N)
    for i in 1:N
        ρ_i = sqrt(r[1, i]^2 + r[2, i]^2)
        ϕ_i = atan(r[2, i], r[1, i])
        z_i = r[3, i]
        e_x, e_y, e_z = electric_guided_mode_profile_cartesian_components(ρ_i, ϕ_i, ϕ₀, f, fiber, polarization_basis)
        d_dot_e = conj(d[i][1]) * e_x + conj(d[i][2]) * e_y + conj(d[i][3]) * e_z
        Ωs[i] = d_dot_e * exp(im * f * fiber.propagation_constant * z_i)
    end
    return Ωs
end

function fill_transmissions_three_level!(t, M, Δes, Δr, Ωs::Number, gs, ω₀, dβ₀, γ)
    for i in eachindex(t)
        t[i] = 1.0 + im * ω₀ * dβ₀ / 2 * gs' * ((M + (-Δes[i] + abs2(Ωs) / (Δes[i] + Δr + im * γ / 2)) * I) \ gs)
    end
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

function transmission_three_level(Δes, fiber, Δr, Ωs::AbstractArray, gs, J, Γ, γ, ζ_ground, ζ_intermediate, ζ_top)
    ω₀ = fiber.frequency
    dβ₀ = fiber.propagation_constant_derivative
    t = zeros(ComplexF64, length(Δes))
    M = -(J + im * Γ / 2)
    fill_transmissions_three_level!(t, M, Δes, Δr, Ωs, gs, ω₀, dβ₀, Γ, γ)
    return t
end

function fill_transmissions_two_level!(t, M, Δes, gs, ω₀, dβ₀)
    for i in eachindex(t)
        t[i] = 1.0 + im * ω₀ * dβ₀ / 2 * gs' * ((M - Δes[i] * I) \ gs)
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
