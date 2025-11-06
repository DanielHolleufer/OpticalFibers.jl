## Vacuum Coefficients ##

"""
    vacuum_coefficients(positions, dipole, ω₀)

Compute the dipole-dipole and decay coefficients for the master equation describing a cloud
of atoms with positions given by the columns in `r` (in cartesian coordinates), dipole
moment `d`, and transition frequency `ω₀` coupled to the vacuum field.
"""
function vacuum_coefficients(positions, dipole, ω₀)
    N = size(positions)[2]
    d_norm2 = sum(abs2, dipole)
    ω₀³ = ω₀^3
    γ₀ = ω₀³ * d_norm2 / (3π)
    coeff = Matrix{ComplexF64}(undef, N, N)
    for j in 1:N
        x_j, y_j, z_j = positions[1, j], positions[2, j], positions[3, j]
        for i in j+1:N
            x_i, y_i, z_i = positions[1, i], positions[2, i], positions[3, i]
            x_ij = x_i - x_j
            y_ij = y_i - y_j
            z_ij = z_i - z_j

            r_norm2 = x_ij^2 + y_ij^2 + z_ij^2
            dr = conj(dipole[1]) * x_ij + conj(dipole[2]) * y_ij + conj(dipole[3]) * z_ij
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


## Guided Mode Coefficients ##

struct GuidedCouplingStrengths
    fwd_rhcp::ComplexF64
    fwd_lhcp::ComplexF64
    bwd_rhcp::ComplexF64
    bwd_lhcp::ComplexF64
end

function GuidedCouplingStrengths(
    fwd_rhcp::Number,
    fwd_lhcp::Number,
    bwd_rhcp::Number,
    bwd_lhcp::Number
)
    return GuidedCouplingStrengths(
        ComplexF64(fwd_rhcp),
        ComplexF64(fwd_lhcp),
        ComplexF64(bwd_rhcp),
        ComplexF64(bwd_lhcp)
    )
end

function Base.show(io::IO, coupling_strengths::GuidedCouplingStrengths)
    println(io, "Coupling strength to different modes:")
    println(io, "Forward RHCP  (f =  1, l =  1): $(coupling_strengths.fwd_rhcp)")
    println(io, "Forward LHCP  (f = -1, l =  1): $(coupling_strengths.fwd_lhcp)")
    println(io, "Backward RHCP (f =  1, l = -1): $(coupling_strengths.bwd_rhcp)")
    print(io,   "Backward LHCP (f = -1, l = -1): $(coupling_strengths.bwd_lhcp)")
end

function guided_coupling_strengths(r, d::Vector{<:Number}, fiber::Fiber)
    x, y, z = r
    ρ = sqrt(x^2 + y^2)
    ϕ = atan(y, x)
    β = fiber.propagation_constant

    e_ρ, e_ϕ, e_z = electric_guided_mode_base(ρ, fiber)

    rhcp_phase_factor = exp(im * ϕ)
    lhcp_phase_factor = conj(rhcp_phase_factor)

    e_ρ_rhcp = e_ρ * rhcp_phase_factor
    e_ρ_lhcp = e_ρ * lhcp_phase_factor
    e_ϕ_rhcp = e_ϕ * rhcp_phase_factor
    e_ϕ_lhcp = -e_ϕ * lhcp_phase_factor

    cosϕ = real(rhcp_phase_factor)
    sinϕ = imag(rhcp_phase_factor)

    e_x_rhcp = e_ρ_rhcp * cosϕ - e_ϕ_rhcp * sinϕ
    e_x_lhcp = e_ρ_lhcp * cosϕ - e_ϕ_lhcp * sinϕ
    e_y_rhcp = e_ρ_rhcp * sinϕ + e_ϕ_rhcp * cosϕ
    e_y_lhcp = e_ρ_lhcp * sinϕ + e_ϕ_lhcp * cosϕ
    e_z_rhcp = e_z * rhcp_phase_factor
    e_z_lhcp = e_z * lhcp_phase_factor

    fwd_phase_factor = exp(im * β * z)
    bwd_phase_factor = conj(fwd_phase_factor)

    g_fwd_rhcp = (d[1] * e_x_rhcp + d[2] * e_y_rhcp + d[3] * e_z_rhcp) * fwd_phase_factor
    g_fwd_lhcp = (d[1] * e_x_lhcp + d[2] * e_y_lhcp + d[3] * e_z_lhcp) * fwd_phase_factor
    g_bwd_rhcp = (d[1] * e_x_rhcp + d[2] * e_y_rhcp - d[3] * e_z_rhcp) * bwd_phase_factor
    g_bwd_lhcp = (d[1] * e_x_lhcp + d[2] * e_y_lhcp - d[3] * e_z_lhcp) * bwd_phase_factor

    return GuidedCouplingStrengths(g_fwd_rhcp, g_fwd_lhcp, g_bwd_rhcp, g_bwd_lhcp)
end

"""
    guided_coefficients(r, d, fiber)

Compute the guided dipole-dipole and decay coefficients for the master equation describing
a cloud of atoms with positions given by the columns in `r` (in cartesian coordinates), and
dipole moment `d` coupled to an optical fiber.
"""
function guided_coefficients(positions, dipole, fiber)
    N = size(positions)[2]
    factor = fiber.frequency * fiber.propagation_constant_derivative / 2
    couplings = Vector{GuidedCouplingStrengths}(undef, N)
    for i in eachindex(couplings)
        couplings[i] = guided_coupling_strengths(view(positions, :, i), dipole, fiber)
    end

    J = Matrix{ComplexF64}(undef, N, N)
    Γ = Matrix{ComplexF64}(undef, N, N)
    guided_coefficients_fill!(J, Γ, positions, factor, couplings)

    return J, Γ
end

function guided_coefficients_fill!(J, Γ, positions, factor, couplings)
    for (j, gj) in enumerate(couplings)
        z_j = positions[3, j]
        for (i, gi) in enumerate(couplings)
            z_i = positions[3, i]

            Γ_ij_fwd = factor * (gi.fwd_rhcp * gj.fwd_rhcp' + gi.fwd_lhcp * gj.fwd_lhcp')
            Γ_ij_bwd = factor * (gi.bwd_rhcp * gj.bwd_rhcp' + gi.bwd_lhcp * gj.bwd_lhcp')

            J[i, j] = im / 2 * (sign(z_i - z_j) * Γ_ij_fwd + sign(z_j - z_i) * Γ_ij_bwd)
            Γ[i, j] = Γ_ij_fwd + Γ_ij_bwd

            J[j, i] = conj(J[i, j])
            Γ[j, i] = conj(Γ[i, j])
        end
    end
end

"""
    guided_directional_coefficients(r, d, fiber)

Compute the guided dipole-dipole and decay coefficients due to the modes with direction `f`
for the master equation describing a cloud of atoms with positions given by the columns in
`r` (in cartesian coordinates), and dipole moment `d` coupled to an optical fiber.
"""
function guided_directional_coefficients(positions, dipole, fiber)
    N = size(positions)[2]
    factor = fiber.frequency * fiber.propagation_constant_derivative / 2
    couplings = Vector{GuidedCouplingStrengths}(undef, N)
    for i in eachindex(couplings)
        couplings[i] = guided_coupling_strengths(view(positions, :, i), dipole, fiber)
    end
    J₊ = Matrix{ComplexF64}(undef, N, N)
    Γ₊ = Matrix{ComplexF64}(undef, N, N)
    J₋ = Matrix{ComplexF64}(undef, N, N)
    Γ₋ = Matrix{ComplexF64}(undef, N, N)
    guided_directional_coefficients_fill!(J₊, Γ₊, J₋, Γ₋, positions, factor, couplings)
    
    return J₊, Γ₊, J₋, Γ₋
end

function guided_directional_coefficients_fill!(
    J₊::Matrix{ComplexF64},
    Γ₊::Matrix{ComplexF64},
    J₋::Matrix{ComplexF64},
    Γ₋::Matrix{ComplexF64},
    positions::Matrix{<:Real},
    factor::Real,
    couplings::Vector{GuidedCouplingStrengths},
)
    for (j, gj) in enumerate(couplings)
        z_j = positions[3, j]
        for (i, gi) in enumerate(couplings)
            z_i = positions[3, i]

            Γ₊[i, j] = factor * (gi.fwd_rhcp * gj.fwd_rhcp' + gi.fwd_lhcp * gj.fwd_lhcp')
            Γ₋[i, j] = factor * (gi.bwd_rhcp * gj.bwd_rhcp' + gi.bwd_lhcp * gj.bwd_lhcp')

            J₊[i, j] = im / 2 * sign(z_i - z_j) * Γ₊[i, j]
            J₋[i, j] = im / 2 * sign(z_j - z_i) * Γ₋[i, j]
            
            J₊[j, i] = conj(J₊[i, j])
            Γ₊[j, i] = conj(Γ₊[i, j])
            J₋[j, i] = conj(J₋[i, j])
            Γ₋[j, i] = conj(Γ₋[i, j])
        end
    end
end


## Radiation Mode Coefficients ##

function radiation_propagation_constants(fiber, resolution)
    ω₀ = fiber.frequency
    n = fiber.refractive_index
    pairs = gauss_legendre_pairs_positive(resolution)
    weights = reverse(weight.(pairs))
    βs = ω₀ * reverse(cos.(pairs))
    hs = sqrt.(n^2 * ω₀^2 .- βs.^2)
    qs = sqrt.(ω₀^2 .- βs.^2)

    return weights, βs, hs, qs
end

function radiation_coupling_strengh_arbitrary_dipole(
    position,
    dipole,
    ω,
    βs,
    qs,
    m_max,
    boundary_coefficients::Array{RadiationBoundaryCoefficients,3},
)
    x, y, z = position
    ρ = sqrt(x^2 + y^2)
    ϕ = atan(y, x)
    cosϕ = cos(ϕ)
    sinϕ = sin(ϕ)
    H1s, dH1s = hankel_evaluations(ρ, m_max, qs)
    Nβ = Int(length(βs) / 2)
    coupling_strengths = Array{ComplexF64,3}(undef, size(boundary_coefficients))
    for (k, l) in enumerate((-1, 1))
        for (j, β) in enumerate(βs)
            j_q = max(Nβ + 1 - j, j - Nβ)
            q = qs[j_q]
            propagation_phase = exp(im * β * z)
            for (i, m) in enumerate(-m_max:m_max)
                i_m = abs(m) + 1
                H = H1s[i_m, j_q]
                dH = dH1s[i_m, j_q]
                angular_phase = exp(im * m * ϕ)
                c = boundary_coefficients[i, j, k]
                eρ, eϕ, ez = electric_radiation_mode_base_external(ρ, ω, β, q, m, H, dH, c)
                ex = eρ * cosϕ - eϕ * sinϕ
                ey = eρ * sinϕ + eϕ * cosϕ
                g = dipole[1] * ex + dipole[2] * ey + dipole[3] * ez
                coupling_strengths[i, j, k] = g * angular_phase * propagation_phase
            end
        end
    end

    return coupling_strengths
end

function radiation_coupling_strengh_transverse_dipole(
    position,
    dipole,
    ω,
    βs,
    qs,
    m_max,
    boundary_coefficients::Array{RadiationBoundaryCoefficients,2},
)
    x, y, z = position
    ρ = sqrt(x^2 + y^2)
    ϕ = atan(y, x)
    cosϕ = cos(ϕ)
    sinϕ = sin(ϕ)
    H1s, dH1s = hankel_evaluations(ρ, m_max, qs)
    Nβ = Int(length(βs) / 2)
    coupling_strengths = Array{ComplexF64,2}(undef, size(boundary_coefficients))
    for (j, β) in enumerate(βs)
        j_q = max(Nβ + 1 - j, j - Nβ)
        q = qs[j_q]
        propagation_phase = exp(im * β * z)
        for (i, m) in enumerate(-m_max:m_max)
            i_m = abs(m) + 1
            H = H1s[i_m, j_q]
            dH = dH1s[i_m, j_q]
            angular_phase = exp(im * m * ϕ)
            c = boundary_coefficients[i, j]
            eρ, eϕ, ez = electric_radiation_mode_base_external(ρ, ω, β, q, m, H, dH, c)
            ex = eρ * cosϕ - eϕ * sinϕ
            ey = eρ * sinϕ + eϕ * cosϕ
            g = dipole[1] * ex + dipole[2] * ey + dipole[3] * ez
            coupling_strengths[i, j] = g * angular_phase * propagation_phase
        end
    end

    return coupling_strengths
end

function radiation_decay_coefficients_arbitrary_dipole(
    positions,
    dipole,
    fiber,
    m_max,
    resolution,
)
    N = size(positions)[2]
    ω₀ = fiber.frequency
    weights, βs, hs, qs = radiation_propagation_constants(fiber, resolution)
    weights = reshape([reverse(weights); weights], 1, 2 * resolution, 1)
    βs = [-reverse(βs); βs]
    boundary = radiation_boundary_coefficients_arbitrary_dipole(m_max, βs, hs, qs, fiber)
    gs = Array{ComplexF64,4}(undef, N, 2 * m_max + 1, 2 * resolution, 2)
    for i in 1:N
        gs[i, :, :, :] = radiation_coupling_strengh_arbitrary_dipole(
            view(positions, :, i),
            dipole,
            ω₀,
            βs,
            qs,
            m_max,
            boundary
        )
    end

    Γ = Matrix{ComplexF64}(undef, N, N)
    for i in 1:N, j in i:N
        Γ[i, j] = sum((gs[i, :, :, :] .* conj(gs[j, :, :, :])) .* weights)
        Γ[j, i] = conj(Γ[i, j])
    end

    return ω₀^2 / 2 * Γ
end

function radiation_decay_coefficients_transverse_dipole(
    positions,
    dipole,
    fiber,
    m_max,
    resolution,
)
    N = size(positions)[2]
    ω₀ = fiber.frequency
    weights, βs, hs, qs = radiation_propagation_constants(fiber, resolution)
    weights = [reverse(weights); weights]
    βs = [-reverse(βs); βs]
    boundary = radiation_boundary_coefficients_transverse_dipole(m_max, βs, hs, qs, fiber)
    gs = Array{ComplexF64,3}(undef, N, 2 * m_max + 1, 2 * resolution)
    for i in 1:N
        gs[i, :, :] = radiation_coupling_strengh_transverse_dipole(
            view(positions, :, i),
            dipole,
            ω₀,
            βs,
            qs,
            m_max,
            boundary
        )
    end

    Γ = Matrix{ComplexF64}(undef, N, N)
    for i in 1:N, j in i:N
        Γ[i, j] = sum((gs[i, :, :] .* conj(gs[j, :, :])) .* weights')
        Γ[j, i] = conj(Γ[i, j])
    end

    return ω₀^2 * Γ
end

function radiation_decay_coefficients(
    positions::Matrix{<:Real},
    dipole::Vector{<:Number},
    fiber::Fiber,
    m_max::Integer,
    resolution::Integer,
)
    if dipole[3] == 0.0
        Γ = radiation_decay_coefficients_transverse_dipole(
            positions,
            dipole,
            fiber,
            m_max,
            resolution
        )
    else
        Γ = radiation_decay_coefficients_arbitrary_dipole(
            positions,
            dipole,
            fiber,
            m_max,
            resolution
        )
    end

    return Γ
end

"""
    radiation_coefficients(positions, dipole, Γ₀, fiber)

Compute the dipole-dipole and decay coefficients for the master equation describing
a cloud of atoms with positions given by the columns in `r` (in cartesian coordinates), and
dipole moment `d` coupled to the radiation modes from an optical fiber.
"""
function radiation_coefficients(
    positions::Matrix{<:Real},
    dipole::Vector{<:Number},
    Γ₀::Real,
    fiber::Fiber;
    m_max::Integer = 20,
    resolution::Integer = 500,
)
    N = size(positions)[2]
    ω₀ = fiber.frequency
    J = Matrix{ComplexF64}(undef, N, N)
    J_vacuum, _ = vacuum_coefficients(positions, dipole, ω₀)
    Γ = radiation_decay_coefficients(positions, dipole, fiber, m_max, resolution)

    for j in 1:N, i in j:N
        J[i, j] = J_vacuum[i, j] * sqrt(Γ[i, i] * Γ[j, j]) / Γ₀
        J[j, i] = J_vacuum[j, i] * sqrt(Γ[i, i] * Γ[j, j]) / Γ₀
    end

    return J, Γ
end
