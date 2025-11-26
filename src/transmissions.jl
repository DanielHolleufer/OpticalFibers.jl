function _loss_function_optim(u, f, xdata, ydata)
    return sum(abs2, ydata .- f.(xdata, u...))
end

function transmission_coefficient(Δ::Real, Δ_shift::Real, Γ_1D::Real, Γ_loss::Real)
    return 1 - im * Γ_1D / (Δ - Δ_shift + im * (Γ_1D + Γ_loss) / 2)
end

function transmission_coefficient(Δ::Real, Γ_1D::Real, Γ_loss::Real, Ω::Real, Δr::Real, γ::Real)
    return 1 - im * Γ_1D / (Δ + im * (Γ_1D + Γ_loss) / 2 - Ω^2 / (Δ + Δr + im * γ / 2))
end

function transmittance(Δ::Real, Γ_1D::Real, Γ_loss::Real)
    return 1 - 4 * Γ_1D * Γ_loss / (4 * Δ^2 + (Γ_1D + Γ_loss)^2)
end

function transmittance(Δ::Real, Γ_1D::Real, Γ_loss::Real, Ω::Real, Δr::Real, γ::Real)
    return abs2(transmission_coefficient_three_level_atom(Δ, Γ_1D, Γ_loss, Ω, Δr, γ))
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

function rabi_frequency(d, positions, field::GuidedField)
    N = size(positions)[2]
    rabis = Vector{ComplexF64}(undef, N)
    for i in 1:N
        x = positions[1, i]
        y = positions[2, i]
        z = positions[3, i]
        ρ = sqrt(x^2 + y^2)
        ϕ = atan(y, x)
        rabis[i] = rabi_frequency(ρ, ϕ, z, d, field)
    end
    return rabis
end

rabi_frequency(d, positions, field::ExternalField) = field.rabi_frequency

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
function transmission_three_level(
    Δes, probe::GuidedMode, atom::ThreeLevelAtom, gs, Ω, H_effective, Γ
)
    ω₀ = frequency(probe)
    dβ₀ = propagation_constant_derivative(probe)
    Δr = atom.upper_detuning
    Γ_er = atom.upper_transition.decay_rate
    coupling = im * ω₀ * dβ₀ / 2
    shift = Δr + im * Γ_er / 2

    M = copy(H_effective)
    t = Vector{ComplexF64}(undef, length(Δes))
    g_temp = similar(gs)
    fill_transmissions_three_level!(t, M, g_temp, Δes, gs, Ω, coupling, shift, Γ)

    return t
end

function fill_transmissions_three_level!(
    t, M::Hessenberg, g_temp, Δes, gs, Ω::Number, coupling, shift, Γ
)
    for i in eachindex(t)
        factor = -Δes[i] + abs2(Ω) / (Δes[i] + shift)
        ldiv!(g_temp, M + factor * I, gs)
        t[i] = 1.0 + coupling * gs' * g_temp
    end
end

function fill_transmissions_three_level!(
    t, M, g_temp, Δes, gs, Ωs::AbstractVector, coupling, shift, Γ
)
    for i in eachindex(t)
        for j in eachindex(Ωs)
            M[j, j] = -Δes[i] + abs2(Ωs[j]) / (Δes[i] + shift) - im * Γ[j, j] / 2
        end
        F = lu(M)
        ldiv!(g_temp, F, gs)
        t[i] = 1.0 + coupling * gs' * g_temp
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
    coupling = im * ω₀ * dβ₀ / 2
    t = zeros(ComplexF64, length(Δes))
    M = hessenberg(-(J + im * Γ / 2))
    g_temp = similar(gs)
    fill_transmissions_two_level!(t, g_temp, M, Δes, gs, coupling)
    return t
end

function fill_transmissions_two_level!(t, g_temp, M, Δes, gs, coupling)
    for i in eachindex(t)
        ldiv!(g_temp, M - Δes[i] * I, gs)
        t[i] = 1.0 + coupling * gs' * g_temp
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
