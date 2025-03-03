"""
    greens_function_free_space(k, r)

Compute the Green's tensor of the electric field in free space with wave number `k` and
position vector `k`.
"""
function greens_function_free_space(k, r)
    rr = r * r'
    kr = k * norm(r)
    G = k * exp(im * kr) / (4π * kr^3) * ((kr^2 + im * kr - 1) * I + (-kr^2 - 3im * kr + 3) * rr / (norm(r)^2))
    return G
end

"""
    vacuum_coefficients(r, d, ω₀)

Compute the dipole-dipole and decay coefficients for the master equation describing a cloud
of atoms with positions given by the columns in `r` (in cartesian coordinates), dipole
moment `d`, and transition frequency `ω₀` coupled to the vacuum field.
"""
function vacuum_coefficients(r, d, ω₀)
    N = size(r)[2]
    γ₀ = ω₀^3 * norm(d)^2 / (3π)
    coeff = zeros(ComplexF64, N, N)
    for j in 1:N
        for i in 1:N
            x_ij = r[1, i] - r[1, j]
            y_ij = r[2, i] - r[2, j]
            z_ij = r[3, i] - r[3, j]
            dr = d[1]' * x_ij + d[2]' * y_ij + d[3]' * z_ij
            kr = ω₀ * sqrt(x_ij^2 + y_ij^2 + z_ij^2)
            coeff[i, j] = ω₀^3 * exp(im * kr) / (4π * kr^3) * ((kr^2 + im * kr - 1) * norm(d)^2 + (-kr^2 - 3im * kr + 3) * abs2(dr) / (kr / ω₀)^2)
        end
    end
    [coeff[i, i] = 0.0 + im * γ₀ / 2 for i in 1:N]
    J = real.(coeff)
    Γ = 2 * imag.(coeff)
    return J, Γ
end

"""
    guided_coupling_strength(ρ, ϕ, z, d, l, f, fiber)

Compute the coupling strength between an atom and a guided fiber mode.

Implementation of Eq. (7), top equation from Fam Le Kien and A. Rauschenbeutel. 
"Nanofiber-mediated chiral radiative coupling between two atoms". Phys. Rev. A 95, 023838
(2017).
"""
function guided_coupling_strength(ρ, ϕ, z, d, l, f, fiber, polarization_basis::CircularPolarization)
    β = fiber.propagation_constant
    ex, ey, ez = electric_mode_components_outside_cartesian(ρ, ϕ, l, f, fiber, polarization_basis)
    de = (d[1] * ex + d[2] * ey + d[3] * ez)
    return de * exp(im * (l * ϕ + f * β * z))
end

"""
    guided_mode_coefficients(r, d, fiber)

Compute the guided dipole-dipole and decay coefficients for the master equation describing
a cloud of atoms with positions given by the columns in `r` (in cartesian coordinates), and
dipole moment `d` coupled to an optical fiber.
"""
function guided_mode_coefficients(r, d, fiber, polarization_basis::CircularPolarization)
    N = size(r)[2]
    ω = fiber.frequency
    dβ = fiber.propagation_constant_derivative
    J = zeros(ComplexF64, N, N)
    Γ = zeros(ComplexF64, N, N)
    guided_mode_coefficients_fill!(J, Γ, r, d, ω, dβ, fiber, polarization_basis::CircularPolarization, N)
    return J, Γ
end

function guided_mode_coefficients_fill!(J, Γ, r, d, ω, dβ, fiber, polarization_basis::CircularPolarization, N)
    for j in 1:N
        for i in 1:N
            J_ij = 0.0
            Γ_ij = 0.0
            ρ_i = sqrt(r[1, i]^2 + r[2, i]^2)
            ρ_j = sqrt(r[1, j]^2 + r[2, j]^2)
            ϕ_i = atan(r[2, i], r[1, i])
            ϕ_j = atan(r[2, j], r[1, j])
            z_i = r[3, i]
            z_j = r[3, j]
            for l in (-1, 1), f in (-1, 1)
                G_i = guided_coupling_strength(ρ_i, ϕ_i, z_i, d, l, f, fiber, polarization_basis::CircularPolarization)
                G_j = guided_coupling_strength(ρ_j, ϕ_j, z_j, d, l, f, fiber, polarization_basis::CircularPolarization)
                J_ij -= sign(f * (z_i - z_j)) * G_i * G_j'
                Γ_ij += G_i * G_j'
            end
            J[i, j] = im * ω * dβ / 4 * J_ij
            Γ[i, j] = ω * dβ / 2 * Γ_ij
        end
    end
end

"""
    guided_mode_directional_coefficients(r, d, fiber)

Compute the guided dipole-dipole and decay coefficients due to the modes with direction `f`
for the master equation describing a cloud of atoms with positions given by the columns in
`r` (in cartesian coordinates), and dipole moment `d` coupled to an optical fiber.
"""
function guided_mode_directional_coefficients(r, d, fiber, f)
    N = size(r)[2]
    ω = fiber.frequency
    dβ = fiber.propagation_constant_derivative
    J = zeros(ComplexF64, N, N)
    Γ = zeros(ComplexF64, N, N)
    guided_mode_directional_coefficients_fill!(J, Γ, r, d, ω, dβ, fiber, polarization_basis::CircularPolarization, N, f)
    return J, Γ
end

function guided_mode_directional_coefficients_fill!(J, Γ, r, d, ω, dβ, fiber, polarization_basis::CircularPolarization, N, f)
    for j in 1:N
        for i in 1:N
            J_ij = 0.0
            Γ_ij = 0.0
            ρ_i = sqrt(r[1, i]^2 + r[2, i]^2)
            ρ_j = sqrt(r[1, j]^2 + r[2, j]^2)
            ϕ_i = atan(r[2, i], r[1, i])
            ϕ_j = atan(r[2, j], r[1, j])
            z_i = r[3, i]
            z_j = r[3, j]
            for l in (-1, 1)
                G_i = guided_coupling_strength(ρ_i, ϕ_i, z_i, d, l, f, fiber, polarization_basis::CircularPolarization)
                G_j = guided_coupling_strength(ρ_j, ϕ_j, z_j, d, l, f, fiber, polarization_basis::CircularPolarization)
                J_ij -= sign(f * (z_i - z_j)) * G_i * G_j'
                Γ_ij += G_i * G_j'
            end
            J[i, j] = im * ω * dβ / 4 * J_ij
            Γ[i, j] = ω * dβ / 2 * Γ_ij
        end
    end
end

"""
    radiative_coupling_strength(ρ, ϕ, z, d, l, f, fiber, polarization_basis::CircularPolarization)

Compute the coupling strength between an atom and a radiation fiber mode.

Implementation of Eq. (7), bottom equation from Fam Le Kien and A. Rauschenbeutel. 
"Nanofiber-mediated chiral radiative coupling between two atoms". Phys. Rev. A 95, 023838
(2017).
"""
function radiative_coupling_strength(ρ, ϕ, z, ω, d, l, m, β, fiber, polarization_basis::CircularPolarization)
    e = electric_radiation_mode(ρ, ϕ, ω, l, m, β, fiber, polarization_basis)
    return sqrt(ω / (4π)) * conj(d)' * e * exp(im * (m * ϕ + β * z))
end

function radiation_mode_decay_coefficients_integrand(β, p)
    ρ_i, ϕ_i, z_i, ρ_j, ϕ_j, z_j, ω, d, l, m, fiber, polarization_basis = p
    G_i = radiative_coupling_strength(ρ_i, ϕ_i, z_i, ω, d, l, m, β, fiber, polarization_basis)
    G_j = radiative_coupling_strength(ρ_j, ϕ_j, z_j, ω, d, l, m, β, fiber, polarization_basis)
    return G_i * conj(G_j)
end

function modes_sum(ρ_i, ϕ_i, z_i, ρ_j, ϕ_j, z_j, ω₀, d, m, fiber, polarization_basis::CircularPolarization, domain)
    Γ_ij = 0.0
    for l in (-1, 1)
        p = (ρ_i, ϕ_i, z_i, ρ_j, ϕ_j, z_j, ω₀, d, l, m, fiber, polarization_basis)
        prob = IntegralProblem(radiation_mode_decay_coefficients_integrand, domain, p)
        sol = solve(prob, HCubatureJL())
        Γ_ij += 2π * sol.u
    end
    return Γ_ij
end

"""
    radiation_mode_decay_coefficients(r, d, fiber, polarization_basis::CircularPolarization; abstol = 1e-3)

Compute the decay coefficients for the master equation describing a cloud of atoms with
positions given by the columns in `r` (in cartesian coordinates), and dipole moment `d`
coupled to the radiation modes from an optical fiber.
"""
function radiation_mode_decay_coefficients(r, d, fiber, polarization_basis::CircularPolarization; abstol = 1e-3)
    N = size(r)[2]
    Γ = zeros(ComplexF64, N, N)
    ω₀ = fiber.frequency
    domain = (-ω₀ + eps(ω₀), ω₀ - eps(ω₀))
    for i in 1:N, j in 1:N
        ρ_i = sqrt(r[1, i]^2 + r[2, i]^2)
        ρ_j = sqrt(r[1, j]^2 + r[2, j]^2)
        ϕ_i = atan(r[2, i], r[1, i])
        ϕ_j = atan(r[2, j], r[1, j])
        z_i = r[3, i]
        z_j = r[3, j]
        m = 0
        Γ_ij = modes_sum(ρ_i, ϕ_i, z_i, ρ_j, ϕ_j, z_j, ω₀, d, m, fiber, polarization_basis, domain)
        while true
            m += 1
            Γ_ij_temp = modes_sum(ρ_i, ϕ_i, z_i, ρ_j, ϕ_j, z_j, ω₀, d, m, fiber, polarization_basis, domain)
            Γ_ij += Γ_ij_temp
            abs(Γ_ij_temp) < abstol && break
        end
        m_cutoff = m
        for m in -m_cutoff:-1
            Γ_ij += modes_sum(ρ_i, ϕ_i, z_i, ρ_j, ϕ_j, z_j, ω₀, d, m, fiber, polarization_basis, domain)
        end
        Γ[i, j] = Γ_ij
    end
    return Γ
end

"""
    radiation_mode_coefficients(r, d, fiber, polarization_basis::CircularPolarization; abstol = 1e-3)

Compute the dipole-dipole and decay coefficients for the master equation describing
a cloud of atoms with positions given by the columns in `r` (in cartesian coordinates), and
dipole moment `d` coupled to the radiation modes from an optical fiber.
"""
function radiation_mode_coefficients(r, d, fiber, polarization_basis::CircularPolarization; abstol = 1e-3)
    N = size(r)[2]
    ω₀ = fiber.frequency
    γ₀ = ω₀^3 / (3π)
    J = zeros(ComplexF64, N, N)
    J_vacuum, _ = vacuum_coefficients(r, d, ω₀)
    Γ = radiation_mode_decay_coefficients(r, d, fiber, polarization_basis; abstol)

    for i in 1:N, j in 1:N
        J[i, j] = J_vacuum[i, j] * sqrt(Γ[i, i] * Γ[j, j]) / γ₀
    end
    
    return J, Γ
end

"""
    radiation_mode_directional_coefficients(r, d, fiber, polarization_basis::CircularPolarization, f; abstol = 1e-3)

Compute the dipole-dipole and decay coefficients for the master equation describing
a cloud of atoms with positions given by the columns in `r` (in cartesian coordinates), and
dipole moment `d` coupled to the radiation modes with directions `f` from an optical fiber.
"""
function radiation_mode_directional_coefficients(r, d, fiber, polarization_basis::CircularPolarization, f; abstol = 1e-3)
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
        Γ_ij = modes_sum(ρ_i, ϕ_i, z_i, ρ_j, ϕ_j, z_j, ω₀, d, m, fiber, polarization_basis, domain)
        while true
            m += 1
            Γ_ij_temp = modes_sum(ρ_i, ϕ_i, z_i, ρ_j, ϕ_j, z_j, ω₀, d, m, fiber, polarization_basis, domain)
            Γ_ij += Γ_ij_temp
            abs(Γ_ij_temp) < abstol && break
        end
        m_cutoff = m
        for m in -m_cutoff:-1
            Γ_ij += modes_sum(ρ_i, ϕ_i, z_i, ρ_j, ϕ_j, z_j, ω₀, d, m, fiber, polarization_basis, domain)
        end
        Γ[i, j] = Γ_ij
    end
    return Γ
end
