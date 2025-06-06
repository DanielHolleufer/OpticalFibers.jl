"""
    vacuum_coefficients(r, d, ω₀)

Compute the dipole-dipole and decay coefficients for the master equation describing a cloud
of atoms with positions given by the columns in `r` (in cartesian coordinates), dipole
moment `d`, and transition frequency `ω₀` coupled to the vacuum field.
"""
function vacuum_coefficients(r, d, ω₀)
    N = size(r)[2]
    d_norm2 = sum(abs2, d)
    ω₀³ = ω₀^3
    γ₀ = ω₀³ * d_norm2 / (3π)
    coeff = zeros(ComplexF64, N, N)
    for j in 1:N
        x_j, y_j, z_j = r[1, j], r[2, j], r[3, j]
        for i in j+1:N
            x_i, y_i, z_i = r[1, i], r[2, i], r[3, i]
            x_ij = x_i - x_j
            y_ij = y_i - y_j
            z_ij = z_i - z_j

            r_norm2 = x_ij^2 + y_ij^2 + z_ij^2
            dr = conj(d[1]) * x_ij + conj(d[2]) * y_ij + conj(d[3]) * z_ij
            kr = ω₀ * sqrt(r_norm2)

            factor = ω₀³ * exp(im * kr) / (4π * kr^3)
            term_1 = (kr^2 + im * kr - 1) * d_norm2
            term_2 = (-kr^2 - 3im * kr + 3) * abs2(dr) / r_norm2
            coeff[i, j] = factor * (term_1 + term_2)
            coeff[j, i] = conj(coeff[i, j])
        end
    end
    coeff[diagind(coeff)] .= im * γ₀ / 2
    J = real.(coeff)
    Γ = 2 * imag.(coeff)
    return J, Γ
end

function vacuum_coefficients2(r, d, ω₀)
    N = size(r)[2]
    d_norm2 = sum(abs2, d)
    ω₀³ = ω₀^3
    γ₀ = ω₀³ * d_norm2 / (3π)
    coeff = zeros(ComplexF64, N, N)
    for j in 1:N
        x_j, y_j, z_j = r[1, j], r[2, j], r[3, j]
        for i in j+1:N
            x_i, y_i, z_i = r[1, i], r[2, i], r[3, i]
            x_ij = x_i - x_j
            y_ij = y_i - y_j
            z_ij = z_i - z_j

            r_norm2 = x_ij^2 + y_ij^2 + z_ij^2
            dr = conj(d[1]) * x_ij + conj(d[2]) * y_ij + conj(d[3]) * z_ij
            kr = ω₀ * sqrt(r_norm2)

            factor = ω₀³ * exp(im * kr) / (4π * kr^3)
            term_1 = (kr^2 + im * kr - 1) * d_norm2
            term_2 = (-kr^2 - 3im * kr + 3) * abs2(dr) / r_norm2
            coeff[i, j] = factor * (term_1 + term_2)
            coeff[j, i] = conj(coeff[i, j])
        end
    end
    coeff[diagind(coeff)] .= im * γ₀ / 2
    J = real.(coeff)
    Γ = 2 * imag.(coeff)
    return J, Γ
end

struct GuidedCouplingStrengths
    forward_counterclockwise::ComplexF64
    forward_clockwise::ComplexF64
    backward_counterclockwise::ComplexF64
    backward_clockwise::ComplexF64
end

function GuidedCouplingStrengths(forward_counterclockwise::Number,
                                 forward_clockwise::Number,
                                 backward_counterclockwise::Number,
                                 backward_clockwise::Number)
    return GuidedCouplingStrengths(ComplexF64(forward_counterclockwise),
                                   ComplexF64(forward_clockwise),
                                   ComplexF64(backward_counterclockwise),
                                   ComplexF64(backward_clockwise))
end

function Base.show(io::IO, coupling_strengths::GuidedCouplingStrengths)
    println(io, "Coupling strength to different modes:")
    println(io, "Forward counterclockwise  (l =  1, f =  1): $(coupling_strengths.forward_counterclockwise)")
    println(io, "Forward clockwise         (l = -1, f =  1): $(coupling_strengths.forward_clockwise)")
    println(io, "Backward counterclockwise (l =  1, f = -1): $(coupling_strengths.backward_counterclockwise)")
    print(io,   "Backward counterclockwise (l = -1, f = -1): $(coupling_strengths.backward_clockwise)")
end

function guided_coupling_strengths(r, d::Vector{<:Number}, fiber::Fiber)
    x, y, z = r
    ρ = sqrt(x^2 + y^2)
    ϕ = atan(y, x)
    β = fiber.propagation_constant

    e_ρ, e_ϕ, e_z = electric_guided_mode_cylindrical_base_components(ρ, fiber)

    exp_iϕ = exp(im * ϕ)
    exp_minus_iϕ = conj(exp_iϕ)

    e_ρ_p = e_ρ * exp_iϕ
    e_ρ_m = e_ρ * exp_minus_iϕ
    e_ϕ_p = e_ϕ * exp_iϕ
    e_ϕ_m = -e_ϕ * exp_minus_iϕ

    cosϕ = real(exp_iϕ)
    sinϕ = imag(exp_iϕ)

    e_x_p = e_ρ_p * cosϕ - e_ϕ_p * sinϕ
    e_x_m = e_ρ_m * cosϕ - e_ϕ_m * sinϕ
    e_y_p = e_ρ_p * sinϕ + e_ϕ_p * cosϕ
    e_y_m = e_ρ_m * sinϕ + e_ϕ_m * cosϕ

    e_z_pp = e_z * exp_iϕ
    e_z_pm = e_z * exp_minus_iϕ
    e_z_mp = -e_z * exp_iϕ
    e_z_mm = -e_z * exp_minus_iϕ

    exp_iβz = exp(im * β * z)
    exp_minus_iβz = conj(exp_iβz)

    de_pp = (d[1] * e_x_p + d[2] * e_y_p + d[3] * e_z_pp) * exp_iβz
    de_pm = (d[1] * e_x_m + d[2] * e_y_m + d[3] * e_z_pm) * exp_iβz
    de_mp = (d[1] * e_x_p + d[2] * e_y_p + d[3] * e_z_mp) * exp_minus_iβz
    de_mm = (d[1] * e_x_m + d[2] * e_y_m + d[3] * e_z_mm) * exp_minus_iβz

    return GuidedCouplingStrengths(de_pp, de_pm, de_mp, de_mm)
end

"""
    guided_mode_coefficients(r, d, fiber)

Compute the guided dipole-dipole and decay coefficients for the master equation describing
a cloud of atoms with positions given by the columns in `r` (in cartesian coordinates), and
dipole moment `d` coupled to an optical fiber.
"""
function guided_mode_coefficients(r, d, fiber)
    N = size(r)[2]
    ω = fiber.frequency
    dβ = fiber.propagation_constant_derivative
    J = zeros(ComplexF64, N, N)
    Γ = zeros(ComplexF64, N, N)
    couplings = [guided_coupling_strengths(r[:, i], d, fiber) for i in 1:N]
    guided_mode_coefficients_fill!(J, Γ, r, ω, dβ, N, couplings)
    return J, Γ
end

function guided_mode_coefficients_fill!(J, Γ, r, ω, dβ, N, couplings)
    for j in 1:N
        z_j = r[3, j]

        for i in j:N
            z_i = r[3, i]

            GGp = (couplings[i].forward_counterclockwise * couplings[j].forward_counterclockwise'
                   + couplings[i].forward_clockwise * couplings[j].forward_clockwise')
            GGm = (couplings[i].backward_counterclockwise * couplings[j].backward_counterclockwise'
                   + couplings[i].backward_clockwise * couplings[j].backward_clockwise')

            J_ij = sign((z_i - z_j)) * GGp + sign(-(z_i - z_j)) * GGm
            Γ_ij = GGp + GGm

            J[i, j] = im * ω * dβ / 4 * J_ij
            Γ[i, j] = ω * dβ / 2 * Γ_ij
            J[j, i] = conj(J[i, j])
            Γ[j, i] = conj(Γ[i, j])
        end
    end
end

"""
    guided_mode_directional_coefficients(r, d, fiber)

Compute the guided dipole-dipole and decay coefficients due to the modes with direction `f`
for the master equation describing a cloud of atoms with positions given by the columns in
`r` (in cartesian coordinates), and dipole moment `d` coupled to an optical fiber.
"""
function guided_mode_directional_coefficients(r, d, fiber)
    N = size(r)[2]
    ω = fiber.frequency
    dβ = fiber.propagation_constant_derivative
    J₊ = zeros(ComplexF64, N, N)
    Γ₊ = zeros(ComplexF64, N, N)
    J₋ = zeros(ComplexF64, N, N)
    Γ₋ = zeros(ComplexF64, N, N)
    couplings = [guided_coupling_strengths(r[:, i], d, fiber) for i in 1:N]
    guided_mode_directional_coefficients_fill!(J₊, Γ₊, J₋, Γ₋, r, ω, dβ, N, couplings)
    return J₊, Γ₊, J₋, Γ₋
end

function guided_mode_directional_coefficients_fill!(J₊, Γ₊, J₋, Γ₋, r, ω, dβ, N, couplings)
    for j in 1:N
        z_j = r[3, j]

        for i in j:N
            z_i = r[3, i]

            GGp = (couplings[i].forward_counterclockwise * couplings[j].forward_counterclockwise'
                   + couplings[i].forward_clockwise * couplings[j].forward_clockwise')
            GGm = (couplings[i].backward_counterclockwise * couplings[j].backward_counterclockwise'
                   + couplings[i].backward_clockwise * couplings[j].backward_clockwise')

            J₊_ij = sign((z_i - z_j)) * GGp
            Γ₊_ij = GGp
            J₋_ij = sign(-(z_i - z_j)) * GGm
            Γ₋_ij = GGm

            J₊[i, j] = im * ω * dβ / 4 * J₊_ij
            Γ₊[i, j] = ω * dβ / 2 * Γ₊_ij
            J₊[j, i] = conj(J₊[i, j])
            Γ₊[j, i] = conj(Γ₊[i, j])

            J₋[i, j] = im * ω * dβ / 4 * J₋_ij
            Γ₋[i, j] = ω * dβ / 2 * Γ₋_ij
            J₋[j, i] = conj(J₋[i, j])
            Γ₋[j, i] = conj(Γ₋[i, j])
        end
    end
end

"""
    radiative_coupling_strength(ρ, ϕ, z, d, l, f, fiber)

Compute the coupling strength between an atom and a radiation fiber mode.

Implementation of Eq. (7), bottom equation from Fam Le Kien and A. Rauschenbeutel. 
"Nanofiber-mediated chiral radiative coupling between two atoms". Phys. Rev. A 95, 023838
(2017).
"""
function radiative_coupling_strength(ρ, ϕ, z, d, ω, β, l, m, fiber)
    e_x, e_y, e_z = electric_radiation_field_cartesian_components(ρ, ϕ, ω, β, l, m, fiber)
    de = d[1] * e_x + d[2] * e_y + d[3] * e_z
    return de * exp(im * (m * ϕ + β * z))
end

function radiation_mode_decay_coefficients_integrand(β, p)
    ρ_i, ϕ_i, z_i, ρ_j, ϕ_j, z_j, ω, d_i, d_j, m, fiber = p
    e_ρ_plus_i, e_ϕ_plus_i, e_z_plus_i, e_ρ_minus_i, e_ϕ_minus_i, e_z_minus_i, e_ρ_plus_j, e_ϕ_plus_j, e_z_plus_j, e_ρ_minus_j, e_ϕ_minus_j, e_z_minus_j, e_ρ_plus_i_r, e_ϕ_plus_i_r, e_z_plus_i_r, e_ρ_minus_i_r, e_ϕ_minus_i_r, e_z_minus_i_r, e_ρ_plus_j_r, e_ϕ_plus_j_r, e_z_plus_j_r, e_ρ_minus_j_r, e_ϕ_minus_j_r, e_z_minus_j_r = electric_radiation_mode_cylindrical_base_components_external_both_polarizations_and_reflected_two_atoms(ρ_i, ρ_j, ω, β, m, fiber)

    e_x_plus_i = e_ρ_plus_i * cos(ϕ_i) - e_ϕ_plus_i * sin(ϕ_i)
    e_y_plus_i = e_ρ_plus_i * sin(ϕ_i) + e_ϕ_plus_i * cos(ϕ_i)
    e_x_minus_i = e_ρ_minus_i * cos(ϕ_i) - e_ϕ_minus_i * sin(ϕ_i)
    e_y_minus_i = e_ρ_minus_i * sin(ϕ_i) + e_ϕ_minus_i * cos(ϕ_i)
    e_x_plus_j = e_ρ_plus_j * cos(ϕ_j) - e_ϕ_plus_j * sin(ϕ_j)
    e_y_plus_j = e_ρ_plus_j * sin(ϕ_j) + e_ϕ_plus_j * cos(ϕ_j)
    e_x_minus_j = e_ρ_minus_j * cos(ϕ_j) - e_ϕ_minus_j * sin(ϕ_j)
    e_y_minus_j = e_ρ_minus_j * sin(ϕ_j) + e_ϕ_minus_j * cos(ϕ_j)

    e_x_plus_i_r = e_ρ_plus_i_r * cos(ϕ_i) - e_ϕ_plus_i_r * sin(ϕ_i)
    e_y_plus_i_r = e_ρ_plus_i_r * sin(ϕ_i) + e_ϕ_plus_i_r * cos(ϕ_i)
    e_x_minus_i_r = e_ρ_minus_i_r * cos(ϕ_i) - e_ϕ_minus_i_r * sin(ϕ_i)
    e_y_minus_i_r = e_ρ_minus_i_r * sin(ϕ_i) + e_ϕ_minus_i_r * cos(ϕ_i)
    e_x_plus_j_r = e_ρ_plus_j_r * cos(ϕ_j) - e_ϕ_plus_j_r * sin(ϕ_j)
    e_y_plus_j_r = e_ρ_plus_j_r * sin(ϕ_j) + e_ϕ_plus_j_r * cos(ϕ_j)
    e_x_minus_j_r = e_ρ_minus_j_r * cos(ϕ_j) - e_ϕ_minus_j_r * sin(ϕ_j)
    e_y_minus_j_r = e_ρ_minus_j_r * sin(ϕ_j) + e_ϕ_minus_j_r * cos(ϕ_j)

    de_plus_i = (d_i[1] * e_x_plus_i + d_i[2] * e_y_plus_i + d_i[3] * e_z_plus_i)
    de_plus_j = (d_j[1] * e_x_plus_j + d_j[2] * e_y_plus_j + d_j[3] * e_z_plus_j)
    de_minus_i = (d_i[1] * e_x_minus_i + d_i[2] * e_y_minus_i + d_i[3] * e_z_minus_i)
    de_minus_j = (d_j[1] * e_x_minus_j + d_j[2] * e_y_minus_j + d_j[3] * e_z_minus_j)

    de_plus_i_r = (d_i[1] * e_x_plus_i_r + d_i[2] * e_y_plus_i_r + d_i[3] * e_z_plus_i_r)
    de_plus_j_r = (d_j[1] * e_x_plus_j_r + d_j[2] * e_y_plus_j_r + d_j[3] * e_z_plus_j_r)
    de_minus_i_r = (d_i[1] * e_x_minus_i_r + d_i[2] * e_y_minus_i_r + d_i[3] * e_z_minus_i_r)
    de_minus_j_r = (d_j[1] * e_x_minus_j_r + d_j[2] * e_y_minus_j_r + d_j[3] * e_z_minus_j_r)

    G_plus = de_plus_i * de_plus_j' * exp(im * (m * (ϕ_i - ϕ_j) + β * (z_i - z_j)))
    G_minus = de_minus_i * de_minus_j' * exp(im * (m * (ϕ_i - ϕ_j) + β * (z_i - z_j)))

    G_plus_r = de_plus_i_r * de_plus_j_r' * exp(im * (m * (ϕ_i - ϕ_j) + β * (z_i - z_j)))
    G_minus_r = de_minus_i_r * de_minus_j_r' * exp(im * (m * (ϕ_i - ϕ_j) + β * (z_i - z_j)))

    return G_plus + G_minus + G_plus_r + G_minus_r
end

function radiation_mode_decay_coefficients_integrand2(β, p)
    ρ_i, ϕ_i, z_i, ρ_j, ϕ_j, z_j, ω, d_i, d_j, m, fiber = p
    e_ρ_plus_i, e_ϕ_plus_i, e_z_plus_i, e_ρ_minus_i, e_ϕ_minus_i, e_z_minus_i, e_ρ_plus_j, e_ϕ_plus_j, e_z_plus_j, e_ρ_minus_j, e_ϕ_minus_j, e_z_minus_j = electric_radiation_mode_cylindrical_base_components_external_both_polarizations_two_atoms(ρ_i, ρ_j, ω, β, m, fiber)

    e_x_plus_i = e_ρ_plus_i * cos(ϕ_i) - e_ϕ_plus_i * sin(ϕ_i)
    e_y_plus_i = e_ρ_plus_i * sin(ϕ_i) + e_ϕ_plus_i * cos(ϕ_i)
    e_x_minus_i = e_ρ_minus_i * cos(ϕ_i) - e_ϕ_minus_i * sin(ϕ_i)
    e_y_minus_i = e_ρ_minus_i * sin(ϕ_i) + e_ϕ_minus_i * cos(ϕ_i)
    e_x_plus_j = e_ρ_plus_j * cos(ϕ_j) - e_ϕ_plus_j * sin(ϕ_j)
    e_y_plus_j = e_ρ_plus_j * sin(ϕ_j) + e_ϕ_plus_j * cos(ϕ_j)
    e_x_minus_j = e_ρ_minus_j * cos(ϕ_j) - e_ϕ_minus_j * sin(ϕ_j)
    e_y_minus_j = e_ρ_minus_j * sin(ϕ_j) + e_ϕ_minus_j * cos(ϕ_j)

    de_plus_i = (d_i[1] * e_x_plus_i + d_i[2] * e_y_plus_i + d_i[3] * e_z_plus_i)
    de_plus_j = (d_j[1] * e_x_plus_j + d_j[2] * e_y_plus_j + d_j[3] * e_z_plus_j)
    de_minus_i = (d_i[1] * e_x_minus_i + d_i[2] * e_y_minus_i + d_i[3] * e_z_minus_i)
    de_minus_j = (d_j[1] * e_x_minus_j + d_j[2] * e_y_minus_j + d_j[3] * e_z_minus_j)

    G_plus = de_plus_i * de_plus_j' * exp(im * (m * (ϕ_i - ϕ_j) + β * (z_i - z_j)))
    G_minus = de_minus_i * de_minus_j' * exp(im * (m * (ϕ_i - ϕ_j) + β * (z_i - z_j)))

    return G_plus + G_minus
end

function radiation_mode_decay_coefficients_integrand_trig(θ, p)
    ρ_i, ϕ_i, z_i, ρ_j, ϕ_j, z_j, ω, d_i, d_j, m, fiber = p
    e_ρ_plus_i, e_ϕ_plus_i, e_z_plus_i, e_ρ_minus_i, e_ϕ_minus_i, e_z_minus_i, e_ρ_plus_j, e_ϕ_plus_j, e_z_plus_j, e_ρ_minus_j, e_ϕ_minus_j, e_z_minus_j = electric_radiation_mode_cylindrical_base_components_external_both_polarizations_two_atoms_trigonometric(ρ_i, ρ_j, ω, θ, m, fiber)

    β = ω * cos(θ)
    q = ω * sin(θ)

    e_x_plus_i = e_ρ_plus_i * cos(ϕ_i) - e_ϕ_plus_i * sin(ϕ_i)
    e_y_plus_i = e_ρ_plus_i * sin(ϕ_i) + e_ϕ_plus_i * cos(ϕ_i)
    e_x_minus_i = e_ρ_minus_i * cos(ϕ_i) - e_ϕ_minus_i * sin(ϕ_i)
    e_y_minus_i = e_ρ_minus_i * sin(ϕ_i) + e_ϕ_minus_i * cos(ϕ_i)
    e_x_plus_j = e_ρ_plus_j * cos(ϕ_j) - e_ϕ_plus_j * sin(ϕ_j)
    e_y_plus_j = e_ρ_plus_j * sin(ϕ_j) + e_ϕ_plus_j * cos(ϕ_j)
    e_x_minus_j = e_ρ_minus_j * cos(ϕ_j) - e_ϕ_minus_j * sin(ϕ_j)
    e_y_minus_j = e_ρ_minus_j * sin(ϕ_j) + e_ϕ_minus_j * cos(ϕ_j)

    de_plus_i = (d_i[1] * e_x_plus_i + d_i[2] * e_y_plus_i + d_i[3] * e_z_plus_i)
    de_plus_j = (d_j[1] * e_x_plus_j + d_j[2] * e_y_plus_j + d_j[3] * e_z_plus_j)
    de_minus_i = (d_i[1] * e_x_minus_i + d_i[2] * e_y_minus_i + d_i[3] * e_z_minus_i)
    de_minus_j = (d_j[1] * e_x_minus_j + d_j[2] * e_y_minus_j + d_j[3] * e_z_minus_j)

    G_plus = de_plus_i * de_plus_j' * exp(im * (m * (ϕ_i - ϕ_j) + β * (z_i - z_j)))
    G_minus = de_minus_i * de_minus_j' * exp(im * (m * (ϕ_i - ϕ_j) + β * (z_i - z_j)))

    return q * (G_plus + G_minus)
end

function radiation_mode_decay_coefficients_integrand2(β, ρ_i, ϕ_i, z_i, ρ_j, ϕ_j, z_j, ω, d_i, d_j, m, fiber)
    β = ω * β
    e_ρ_plus_i, e_ϕ_plus_i, e_z_plus_i, e_ρ_minus_i, e_ϕ_minus_i, e_z_minus_i, e_ρ_plus_j, e_ϕ_plus_j, e_z_plus_j, e_ρ_minus_j, e_ϕ_minus_j, e_z_minus_j = electric_radiation_mode_cylindrical_base_components_external_both_polarizations_two_atoms(ρ_i, ρ_j, ω, β, m, fiber)

    e_x_plus_i = e_ρ_plus_i * cos(ϕ_i) - e_ϕ_plus_i * sin(ϕ_i)
    e_y_plus_i = e_ρ_plus_i * sin(ϕ_i) + e_ϕ_plus_i * cos(ϕ_i)
    e_x_minus_i = e_ρ_minus_i * cos(ϕ_i) - e_ϕ_minus_i * sin(ϕ_i)
    e_y_minus_i = e_ρ_minus_i * sin(ϕ_i) + e_ϕ_minus_i * cos(ϕ_i)
    e_x_plus_j = e_ρ_plus_j * cos(ϕ_j) - e_ϕ_plus_j * sin(ϕ_j)
    e_y_plus_j = e_ρ_plus_j * sin(ϕ_j) + e_ϕ_plus_j * cos(ϕ_j)
    e_x_minus_j = e_ρ_minus_j * cos(ϕ_j) - e_ϕ_minus_j * sin(ϕ_j)
    e_y_minus_j = e_ρ_minus_j * sin(ϕ_j) + e_ϕ_minus_j * cos(ϕ_j)

    de_plus_i = (d_i[1] * e_x_plus_i + d_i[2] * e_y_plus_i + d_i[3] * e_z_plus_i)
    de_plus_j = (d_j[1] * e_x_plus_j + d_j[2] * e_y_plus_j + d_j[3] * e_z_plus_j)
    de_minus_i = (d_i[1] * e_x_minus_i + d_i[2] * e_y_minus_i + d_i[3] * e_z_minus_i)
    de_minus_j = (d_j[1] * e_x_minus_j + d_j[2] * e_y_minus_j + d_j[3] * e_z_minus_j)

    G_plus = de_plus_i * de_plus_j' * exp(im * (m * (ϕ_i - ϕ_j) + β * (z_i - z_j)))
    G_minus = de_minus_i * de_minus_j' * exp(im * (m * (ϕ_i - ϕ_j) + β * (z_i - z_j)))

    return G_plus + G_minus
end

function modes_sum_l(ρ_i, ϕ_i, z_i, ρ_j, ϕ_j, z_j, ω₀, d_i, d_j, m, fiber, domain)
    p = (ρ_i, ϕ_i, z_i, ρ_j, ϕ_j, z_j, ω₀, d_i, d_j, m, fiber)
    prob = IntegralProblem(radiation_mode_decay_coefficients_integrand, domain, p)
    sol = solve(prob, QuadGKJL())
    return ω₀ / 2 * sol.u
end

function modes_sum_trig(ρ_i, ϕ_i, z_i, ρ_j, ϕ_j, z_j, ω₀, d_i, d_j, m, fiber)
    p = (ρ_i, ϕ_i, z_i, ρ_j, ϕ_j, z_j, ω₀, d_i, d_j, m, fiber)
    prob = IntegralProblem(radiation_mode_decay_coefficients_integrand_trig, (1e-5, π - 1e-5), p)
    sol = solve(prob, QuadGKJL())
    return ω₀ / 2 * sol.u
end

function modes_sum_alt(ρ_i, ϕ_i, z_i, ρ_j, ϕ_j, z_j, ω₀, d_i, d_j, m, fiber)
    resolution = 41
    xs = zeros(resolution)
    ws = zeros(resolution)
    for l in 1:resolution
        θ, w = gauss_legendre_pair(resolution, l)
        xs[l] = cos(θ)
        ws[l] = w
    end
    vals = [radiation_mode_decay_coefficients_integrand2(β, ρ_i, ϕ_i, z_i, ρ_j, ϕ_j, z_j, ω₀, d_i, d_j, m, fiber) for β in xs]
    sol = sum(vals .* ws)

    return ω₀^2 / 2 * sol
end

function modes_sum_trapez(ρ_i, ϕ_i, z_i, ρ_j, ϕ_j, z_j, ω₀, d_i, d_j, m, fiber, n)
    xs = LinRange(-1, 1, n)
    Δx = xs[2] - xs[1]
    vals = [radiation_mode_decay_coefficients_integrand2(β, ρ_i, ϕ_i, z_i, ρ_j, ϕ_j, z_j, ω₀, d_i, d_j, m, fiber) for β in xs]
    result = 0.0
    for i in 2:(n-2)
        result += (vals[i] + vals[i+1]) / 2 * Δx
    end
    return ω₀^2 / 2 * result
end

function radiation_mode_decay_coefficients_integrand_old(β, p)
    ρ_i, ϕ_i, z_i, ρ_j, ϕ_j, z_j, ω, d_i, d_j, l, m, fiber = p
    G_i = radiative_coupling_strength(ρ_i, ϕ_i, z_i, d_i, ω, β, l, m, fiber)
    G_j = radiative_coupling_strength(ρ_j, ϕ_j, z_j, d_j, ω, β, l, m, fiber)
    return G_i * conj(G_j)
end

function modes_sum_old(ρ_i, ϕ_i, z_i, ρ_j, ϕ_j, z_j, ω₀, d_i, d_j, m, fiber, domain)
    Γ_ij = 0.0
    for l in (-1, 1)
        p = (ρ_i, ϕ_i, z_i, ρ_j, ϕ_j, z_j, ω₀, d_i, d_j, l, m, fiber)
        prob = IntegralProblem(radiation_mode_decay_coefficients_integrand_old, domain, p)
        sol = solve(prob, QuadGKJL())
        Γ_ij += sol.u
    end
    return ω₀ / 2 * Γ_ij
end

function is_series_converged(partial_sum, latest_term, index, abstol, reltol, maxiter)
    abs(latest_term) < abstol && return true
    abs(latest_term / partial_sum) < reltol && return true
    if index == maxiter
        println("Warning: series did not converge after $maxiter iterations.")
        println("Latest term: $latest_term")
        println("Latest term / partial sum: $(abs(latest_term / partial_sum))\n")
        return true
    end
    return false
end

"""
    radiation_mode_decay_coefficients(r, d, fiber; abstol = 1e-3)

Compute the decay coefficients for the master equation describing a cloud of atoms with
positions given by the columns in `r` (in cartesian coordinates), and dipole moment `d`
coupled to the radiation modes from an optical fiber.
"""
function radiation_mode_decay_coefficients(r, d, Γ₀, fiber; abstol=1e-6, reltol=1e-6, maxiter=50)
    N = size(r)[2]
    Γ = zeros(ComplexF64, N, N)
    ω₀ = fiber.frequency
    domain = (-ω₀ + 1e-3 * ω₀, ω₀ - 1e-3 * ω₀)
    ρ = sqrt.(r[1, :] .^ 2 .+ r[2, :] .^ 2)
    ϕ = atan.(r[2, :], r[1, :])
    z = r[3, :]
    for j in 1:N
        for i in j:N
            m = 0
            Γ_ij = modes_sum_old(ρ[i], ϕ[i], z[i], ρ[j], ϕ[j], z[j], ω₀, d, d, m, fiber, domain)
            while true
                m += 1
                Γ_ij_temp = modes_sum_old(ρ[i], ϕ[i], z[i], ρ[j], ϕ[j], z[j], ω₀, d, d, m, fiber, domain)
                Γ_ij += Γ_ij_temp
                is_series_converged(Γ_ij / Γ₀, Γ_ij_temp / Γ₀, m, abstol, reltol, maxiter) && break
            end
            Γ[i, j] = Γ_ij
            Γ[j, i] = conj(Γ_ij)
        end
    end
    return Γ
end


function radiation_mode_decay_coefficients2(r, d, Γ₀, fiber; abstol=1e-6, reltol=1e-6, maxiter=50)
    N = size(r)[2]
    Γ = zeros(ComplexF64, N, N)
    ω₀ = fiber.frequency
    domain = (-ω₀ + 1e-3 * ω₀, ω₀ - 1e-3 * ω₀)
    ρ = sqrt.(r[1, :] .^ 2 .+ r[2, :] .^ 2)
    ϕ = atan.(r[2, :], r[1, :])
    z = r[3, :]
    for j in 1:N
        for i in j:N
            m = 0
            Γ_ij = modes_sum(ρ[i], ϕ[i], z[i], ρ[j], ϕ[j], z[j], ω₀, d, d, m, fiber, domain)
            while true
                m += 1
                Γ_ij_temp = modes_sum(ρ[i], ϕ[i], z[i], ρ[j], ϕ[j], z[j], ω₀, d, d, m, fiber, domain)
                Γ_ij += Γ_ij_temp
                is_series_converged(Γ_ij / Γ₀, Γ_ij_temp / Γ₀, m, abstol, reltol, maxiter) && break
            end
            m_cutoff = m
            for m in -m_cutoff:-1
                Γ_ij += modes_sum(ρ[i], ϕ[i], z[i], ρ[j], ϕ[j], z[j], ω₀, d, d, m, fiber, domain)
            end
            Γ[i, j] = Γ_ij
            Γ[j, i] = conj(Γ_ij)
        end
    end
    return Γ
end


function radiation_mode_decay_coefficients_old(r, d, fiber; abstol=1e-6, maxiter=50)
    N = size(r)[2]
    Γ = zeros(ComplexF64, N, N)
    ω₀ = fiber.frequency
    domain = (-ω₀ + ω₀ * 1e-6, ω₀ - ω₀ * 1e-6)
    ρ = sqrt.(r[1, :] .^ 2 .+ r[2, :] .^ 2)
    ϕ = atan.(r[2, :], r[1, :])
    z = r[3, :]
    for j in 1:N
        for i in j:N
            m = 0
            Γ_ij = modes_sum_old(ρ[i], ϕ[i], z[i], ρ[j], ϕ[j], z[j], ω₀, d, d, m, fiber, domain)
            while true
                m += 1
                m > maxiter && break
                Γ_ij_temp = modes_sum_old(ρ[i], ϕ[i], z[i], ρ[j], ϕ[j], z[j], ω₀, d, d, m, fiber, domain)
                Γ_ij += Γ_ij_temp
                abs(Γ_ij_temp) < abstol && break
            end
            m_cutoff = m
            for m in -m_cutoff:-1
                Γ_ij += modes_sum_old(ρ[i], ϕ[i], z[i], ρ[j], ϕ[j], z[j], ω₀, d, d, m, fiber, domain)
            end
            Γ[i, j] = Γ_ij
            Γ[j, i] = Γ_ij'
        end
    end
    return Γ
end

"""
    radiation_mode_coefficients(r, d, fiber; abstol = 1e-6)

Compute the dipole-dipole and decay coefficients for the master equation describing
a cloud of atoms with positions given by the columns in `r` (in cartesian coordinates), and
dipole moment `d` coupled to the radiation modes from an optical fiber.
"""
function radiation_mode_coefficients(r, d, Γ₀, fiber; abstol=1e-6, reltol=1e-6, maxiter=50)
    N = size(r)[2]
    ω₀ = fiber.frequency
    J = zeros(ComplexF64, N, N)
    J_vacuum, _ = vacuum_coefficients(r, d, ω₀)
    Γ = radiation_mode_decay_coefficients(r, d, Γ₀, fiber; abstol=abstol, reltol=reltol, maxiter=maxiter)

    for j in 1:N
        for i in j+1:N
            J[i, j] = J_vacuum[i, j] * sqrt(Γ[i, i] * Γ[j, j]) / Γ₀
            J[j, i] = conj(J[i, j])
        end
    end

    return J, Γ
end

function radiation_mode_coefficients_old(r, d, Γ₀, fiber; abstol=1e-6, maxiter=50)
    N = size(r)[2]
    ω₀ = fiber.frequency
    J = zeros(ComplexF64, N, N)
    J_vacuum, _ = vacuum_coefficients(r, d, ω₀)
    Γ = radiation_mode_decay_coefficients_old(r, d, fiber; abstol=abstol, maxiter=maxiter)

    for j in 1:N
        for i in j:N
            J[i, j] = J_vacuum[i, j] * sqrt(Γ[i, i] * Γ[j, j]) / Γ₀
            J[j, i] = J_vacuum[j, i] * sqrt(Γ[i, i] * Γ[j, j]) / Γ₀
        end
    end

    return J, Γ
end

"""
    radiation_mode_directional_coefficients(r, d, fiber, f; abstol = 1e-6)

Compute the dipole-dipole and decay coefficients for the master equation describing
a cloud of atoms with positions given by the columns in `r` (in cartesian coordinates), and
dipole moment `d` coupled to the radiation modes with directions `f` from an optical fiber.
"""
function radiation_mode_directional_coefficients(r, d, fiber, f; abstol=1e-6)
    N = size(r)[2]
    Γ = zeros(ComplexF64, N, N)
    ω₀ = fiber.frequency

    if f == 1
        domain = (0.0, ω₀ - eps(ω₀))
    elseif f == -1
        domain = (-ω₀ + eps(ω₀), 0.0)
    else
        throw(DomainError(f, "Directional parameter must be either +1 or -1."))
    end

    for i in 1:N, j in 1:N
        ρ_i = sqrt(r[1, i]^2 + r[2, i]^2)
        ρ_j = sqrt(r[1, j]^2 + r[2, j]^2)
        ϕ_i = atan(r[2, i], r[1, i])
        ϕ_j = atan(r[2, j], r[1, j])
        z_i = r[3, i]
        z_j = r[3, j]
        m = 0
        Γ_ij = modes_sum(ρ_i, ϕ_i, z_i, ρ_j, ϕ_j, z_j, ω₀, d, d, m, fiber, domain)
        while true
            m += 1
            Γ_ij_temp = modes_sum(ρ_i, ϕ_i, z_i, ρ_j, ϕ_j, z_j, ω₀, d, d, m, fiber, domain)
            Γ_ij += Γ_ij_temp
            abs(Γ_ij_temp) < abstol && break
        end
        m_cutoff = m
        for m in -m_cutoff:-1
            Γ_ij += modes_sum(ρ_i, ϕ_i, z_i, ρ_j, ϕ_j, z_j, ω₀, d, d, m, fiber, domain)
        end
        Γ[i, j] = Γ_ij
    end
    return Γ
end
