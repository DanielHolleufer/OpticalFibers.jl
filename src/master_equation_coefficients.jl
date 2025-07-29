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
    coeff = Matrix{ComplexF64}(undef, N, N)
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

struct CouplingStrengths
    f_plus_l_plus::ComplexF64
    f_plus_l_minus::ComplexF64
    f_minus_l_plus::ComplexF64
    f_minus_l_minus::ComplexF64
end

function CouplingStrengths(f_plus_l_plus::Number, f_plus_l_minus::Number,
                           f_minus_l_plus::Number, f_minus_l_minus::Number)
    return CouplingStrengths(ComplexF64(f_plus_l_plus), ComplexF64(f_plus_l_minus),
                             ComplexF64(f_minus_l_plus), ComplexF64(f_minus_l_minus))
end

function Base.show(io::IO, coupling_strengths::CouplingStrengths)
    println(io, "Coupling strength to different modes:")
    println(io, "(f =  1, l =  1): $(coupling_strengths.f_plus_l_plus)")
    println(io, "(f = -1, l =  1): $(coupling_strengths.f_plus_l_minus)")
    println(io, "(f =  1, l = -1): $(coupling_strengths.f_minus_l_plus)")
    print(io,   "(f = -1, l = -1): $(coupling_strengths.f_minus_l_minus)")
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

    return CouplingStrengths(de_pp, de_pm, de_mp, de_mm)
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

            GGp = (couplings[i].f_plus_l_plus * couplings[j].f_plus_l_plus'
                   + couplings[i].f_plus_l_minus * couplings[j].f_plus_l_minus')
            GGm = (couplings[i].f_minus_l_plus * couplings[j].f_minus_l_plus'
                   + couplings[i].f_minus_l_minus * couplings[j].f_minus_l_minus')

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

            GGp = (couplings[i].f_plus_l_plus * couplings[j].f_plus_l_plus'
                   + couplings[i].f_plus_l_minus * couplings[j].f_plus_l_minus')
            GGm = (couplings[i].f_minus_l_plus * couplings[j].f_minus_l_plus'
                   + couplings[i].f_minus_l_minus * couplings[j].f_minus_l_minus')

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

function radiative_coupling_strengths(β, p)
    ρ_i, ϕ_i, z_i, ρ_j, ϕ_j, z_j, ω, d_i, d_j, m, fiber = p

    e_ρ_p_i, e_y_ϕ_p_i, e_z_p_i, 
    e_ρ_m_i, e_y_ϕ_m_i, e_z_m_i, 
    e_ρ_p_j, e_y_ϕ_p_j, e_z_p_j,
    e_ρ_m_j, e_y_ϕ_m_j, e_z_m_j = electric_radiation_mode_cylindrical_base_components_external_both_polarizations_and_reflected_two_atoms(ρ_i, ρ_j, ω, β, m, fiber)

    c_i = cos(ϕ_i)
    s_i = sin(ϕ_i)
    c_j = cos(ϕ_j)
    s_j = sin(ϕ_j)

    e_x_p_i = e_ρ_p_i * c_i - e_y_ϕ_p_i * s_i
    e_y_p_i = e_ρ_p_i * s_i + e_y_ϕ_p_i * c_i
    e_x_m_i = e_ρ_m_i * c_i - e_y_ϕ_m_i * s_i
    e_y_m_i = e_ρ_m_i * s_i + e_y_ϕ_m_i * c_i
    e_x_p_j = e_ρ_p_j * c_j - e_y_ϕ_p_j * s_j
    e_y_p_j = e_ρ_p_j * s_j + e_y_ϕ_p_j * c_j
    e_x_m_j = e_ρ_m_j * c_j - e_y_ϕ_m_j * s_j
    e_y_m_j = e_ρ_m_j * s_j + e_y_ϕ_m_j * c_j

    de_p_i = (d_i[1] * e_x_p_i + d_i[2] * e_y_p_i + d_i[3] * e_z_p_i) * exp(im * (m * ϕ_i + β * z_i))
    de_p_j = (d_j[1] * e_x_p_j + d_j[2] * e_y_p_j + d_j[3] * e_z_p_j) * exp(im * (m * ϕ_j + β * z_j))
    de_m_i = (d_i[1] * e_x_m_i + d_i[2] * e_y_m_i + d_i[3] * e_z_m_i) * exp(im * (m * ϕ_i + β * z_i))
    de_m_j = (d_j[1] * e_x_m_j + d_j[2] * e_y_m_j + d_j[3] * e_z_m_j) * exp(im * (m * ϕ_j + β * z_j))

    return de_p_i * de_p_j' + de_m_i * de_m_j'
end

function radiative_coupling_strengths_plus_minus(β, p)
    ρ_i, ϕ_i, z_i, ρ_j, ϕ_j, z_j, ω, d_i, d_j, m, fiber = p
    e_ρ_p_i, e_y_ϕ_p_i, e_z_p_i, 
    e_ρ_m_i, e_y_ϕ_m_i, e_z_m_i, 
    e_ρ_p_j, e_y_ϕ_p_j, e_z_p_j,
    e_ρ_m_j, e_y_ϕ_m_j, e_z_m_j, 
    e_ρ_p_i_r, e_y_ϕ_p_i_r, e_z_p_i_r,
    e_ρ_m_i_r, e_y_ϕ_m_i_r, e_z_m_i_r,
    e_ρ_p_j_r, e_y_ϕ_p_j_r, e_z_p_j_r,
    e_ρ_m_j_r, e_y_ϕ_m_j_r, e_z_m_j_r = electric_radiation_mode_cylindrical_base_components_external_both_polarizations_and_reflected_two_atoms(ρ_i, ρ_j, ω, β, m, fiber)

    c_i = cos(ϕ_i)
    s_i = sin(ϕ_i)
    c_j = cos(ϕ_j)
    s_j = sin(ϕ_j)

    e_x_p_i = e_ρ_p_i * c_i - e_y_ϕ_p_i * s_i
    e_y_p_i = e_ρ_p_i * s_i + e_y_ϕ_p_i * c_i
    e_x_m_i = e_ρ_m_i * c_i - e_y_ϕ_m_i * s_i
    e_y_m_i = e_ρ_m_i * s_i + e_y_ϕ_m_i * c_i
    e_x_p_j = e_ρ_p_j * c_j - e_y_ϕ_p_j * s_j
    e_y_p_j = e_ρ_p_j * s_j + e_y_ϕ_p_j * c_j
    e_x_m_j = e_ρ_m_j * c_j - e_y_ϕ_m_j * s_j
    e_y_m_j = e_ρ_m_j * s_j + e_y_ϕ_m_j * c_j
    e_x_p_i_r = e_ρ_p_i_r * c_i - e_y_ϕ_p_i_r * s_i
    e_y_p_i_r = e_ρ_p_i_r * s_i + e_y_ϕ_p_i_r * c_i
    e_x_m_i_r = e_ρ_m_i_r * c_i - e_y_ϕ_m_i_r * s_i
    e_y_m_i_r = e_ρ_m_i_r * s_i + e_y_ϕ_m_i_r * c_i
    e_x_p_j_r = e_ρ_p_j_r * c_j - e_y_ϕ_p_j_r * s_j
    e_y_p_j_r = e_ρ_p_j_r * s_j + e_y_ϕ_p_j_r * c_j
    e_x_m_j_r = e_ρ_m_j_r * c_j - e_y_ϕ_m_j_r * s_j
    e_y_m_j_r = e_ρ_m_j_r * s_j + e_y_ϕ_m_j_r * c_j

    de_p_i = (d_i[1] * e_x_p_i + d_i[2] * e_y_p_i + d_i[3] * e_z_p_i) * exp(im * (m * ϕ_i + β * z_i))
    de_p_j = (d_j[1] * e_x_p_j + d_j[2] * e_y_p_j + d_j[3] * e_z_p_j) * exp(im * (m * ϕ_j + β * z_j))
    de_m_i = (d_i[1] * e_x_m_i + d_i[2] * e_y_m_i + d_i[3] * e_z_m_i) * exp(im * (m * ϕ_i + β * z_i))
    de_m_j = (d_j[1] * e_x_m_j + d_j[2] * e_y_m_j + d_j[3] * e_z_m_j) * exp(im * (m * ϕ_j + β * z_j))
    de_p_i_r = (d_i[1] * e_x_p_i_r + d_i[2] * e_y_p_i_r + d_i[3] * e_z_p_i_r) * exp(im * (-m * ϕ_i + β * z_i))
    de_p_j_r = (d_j[1] * e_x_p_j_r + d_j[2] * e_y_p_j_r + d_j[3] * e_z_p_j_r) * exp(im * (-m * ϕ_j + β * z_j))
    de_m_i_r = (d_i[1] * e_x_m_i_r + d_i[2] * e_y_m_i_r + d_i[3] * e_z_m_i_r) * exp(im * (-m * ϕ_i + β * z_i))
    de_m_j_r = (d_j[1] * e_x_m_j_r + d_j[2] * e_y_m_j_r + d_j[3] * e_z_m_j_r) * exp(im * (-m * ϕ_j + β * z_j))

    return de_p_i * de_p_j' + de_m_i * de_m_j' + de_p_i_r * de_p_j_r' + de_m_i_r * de_m_j_r'
end

function modes_sum(ρ_i, ϕ_i, z_i, ρ_j, ϕ_j, z_j, ω₀, d_i, d_j, m, fiber, domain)
    p = (ρ_i, ϕ_i, z_i, ρ_j, ϕ_j, z_j, ω₀, d_i, d_j, m, fiber)
    prob = IntegralProblem(radiative_coupling_strengths, domain, p)
    sol = solve(prob, QuadGKJL())
    return ω₀ / 2 * sol.u
end

function modes_sum_plus_minus(ρ_i, ϕ_i, z_i, ρ_j, ϕ_j, z_j, ω₀, d_i, d_j, m, fiber, domain)
    p = (ρ_i, ϕ_i, z_i, ρ_j, ϕ_j, z_j, ω₀, d_i, d_j, m, fiber)
    prob = IntegralProblem(radiative_coupling_strengths_plus_minus, domain, p)
    sol = solve(prob, QuadGKJL())
    return ω₀ / 2 * sol.u
end

function has_series_converged(partial_sum, latest_term, index, abstol, reltol, maxiter;
                              disable_warnings=false)
    abs(latest_term) < abstol && return true
    abs(latest_term / partial_sum) < reltol && return true
    if index == maxiter
        disable_warnings || println("Warning: series did not converge after $maxiter iterations.")
        disable_warnings || println("Latest term: $latest_term")
        disable_warnings || println("Latest term / partial sum: $(abs(latest_term / partial_sum))\n")
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
function radiation_mode_decay_coefficients(r, d, Γ₀, fiber, abstol, reltol, maxiter, disable_warnings)
    N = size(r)[2]
    Γ = Matrix{ComplexF64}(undef, N, N)
    ω₀ = fiber.frequency
    domain = (-ω₀ + ω₀ * 1e-6, ω₀ - ω₀ * 1e-6)
    ρ = sqrt.(r[1, :] .^ 2 .+ r[2, :] .^ 2)
    ϕ = atan.(r[2, :], r[1, :])
    z = r[3, :]
    for j in 1:N
        for i in j:N
            m = 0
            Γ_ij = modes_sum(ρ[i], ϕ[i], z[i], ρ[j], ϕ[j], z[j], ω₀, d, d, m, fiber, domain)
            has_converged = false
            while !has_converged
                m += 1
                Γ_ij_temp = modes_sum_plus_minus(ρ[i], ϕ[i], z[i], ρ[j], ϕ[j], z[j], ω₀, d, d, m, fiber, domain)
                Γ_ij += Γ_ij_temp
                has_converged = has_series_converged(Γ_ij / Γ₀, Γ_ij_temp / Γ₀, m, abstol, reltol, maxiter; disable_warnings)
            end
            Γ[i, j] = Γ_ij
            Γ[j, i] = Γ_ij'
        end
    end

    return Γ
end

function radiation_mode_decay_coefficients_new(r, d, Γ₀, fiber, abstol, reltol, maxiter, resolution, disable_warnings)
    N = size(r)[2]
    Γ = Matrix{ComplexF64}(undef, N, N)
    a = fiber.radius
    ω₀ = fiber.frequency
    n = fiber.refractive_index

    ρ = sqrt.(r[1, :] .^ 2 .+ r[2, :] .^ 2)
    ϕ = atan.(r[2, :], r[1, :])
    z = r[3, :]

    # METHOD (WiP):
    # - Define m values from 0 to some m_max.
    # - Compute Gauss-Legendre pairs and get β values.
    # - Compute h and q values.
    # - Compute all bessel and hankel functions at every β and m in (-1, m_max + 1). 
    # - Compute all surface, auxillary, and boundary coefficients using the computed
    #   bessel and hankel functions, and h and q values.
    # - Compute all hankel coefficients for every atom.
    # - Compute the coupling strengths for every atom.
    # - Perform Gauss-Legendre quadrature for each pair of atoms.

    m_max = 10
    ms = 0:m_max

    pairs = gauss_legendre_pairs(resolution)
    βs = ω₀ * cos.(pairs)

    hs = sqrt.(n^2 * ω₀^2 .- βs.^2)
    qs = sqrt.(ω₀^2 .- β.^2)

    
    
    

    dJas = Matrix{Float64}(undef, m_max + 1, resolution)
    dh₁as = Matrix{ComplexF64}(undef, m_max + 1, resolution)
    dJas[1, :] = 
    for m in 1:m_max
        for k in 1:resolution
            dJas[m + 1, k] = 0.5 * (Jas[m, k] - Jas[m + 2, k])
            dh₁as[m + 1, k] = hankel_h1(m, qs[k] * a)
        end
    end


    surface_coefficients = Matrix{RadiationSurfaceCoefficients}(undef, m_max, resolution)
    auxillary_coefficients = Matrix{RadiationAuxillaryCoefficients}(undef, m_max, resolution)
    boundary_coefficients = Matrix{RadiationBoundaryCoefficients}(undef, m_max, resolution)

    for m in ms
        for (k, β) in enumerate(βs)
            surface_coefficients[m, k] = radiation_surface_coefficients_efficient(a, m, h[k], q[k])
            auxillary_coefficients[m, k] = radiation_auxillary_coefficients(ω₀, β, m, h[k], q[k], fiber)
            boundary_coefficients[m, k] = radiation_boundary_coefficients(β, a, ω₀, n, m)
        end
    end

    for j in 1:N
        for i in j:N
            m = 0
            Γ_ij = modes_sum(ρ[i], ϕ[i], z[i], ρ[j], ϕ[j], z[j], ω₀, d, d, m, fiber, domain)
            has_converged = false
            while !has_converged
                m += 1
                Γ_ij_temp = modes_sum_plus_minus(ρ[i], ϕ[i], z[i], ρ[j], ϕ[j], z[j], ω₀, d, d, m, fiber, domain)
                Γ_ij += Γ_ij_temp
                has_converged = has_series_converged(Γ_ij / Γ₀, Γ_ij_temp / Γ₀, m, abstol, reltol, maxiter; disable_warnings)
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
function radiation_mode_coefficients(r, d, Γ₀, fiber; abstol=1e-6, reltol=1e-6, maxiter=50,
                                     disable_warnings=false)
    N = size(r)[2]
    ω₀ = fiber.frequency
    J = Matrix{ComplexF64}(undef, N, N)
    J_vacuum, _ = vacuum_coefficients(r, d, ω₀)
    Γ = radiation_mode_decay_coefficients(r, d, Γ₀, fiber, abstol, reltol, maxiter, disable_warnings)

    for j in 1:N
        for i in j:N
            J[i, j] = J_vacuum[i, j] * sqrt(Γ[i, i] * Γ[j, j]) / Γ₀
            J[j, i] = J_vacuum[j, i] * sqrt(Γ[i, i] * Γ[j, j]) / Γ₀
        end
    end

    return J, Γ
end

function radiation_mode_coefficients_new(r, d, Γ₀, fiber; abstol=1e-6, reltol=1e-6, maxiter=50,
                                         resolution=512, disable_warnings=false)
    N = size(r)[2]
    ω₀ = fiber.frequency
    J = Matrix{ComplexF64}(undef, N, N)
    J_vacuum, _ = vacuum_coefficients(r, d, ω₀)
    Γ = radiation_mode_decay_coefficients(r, d, Γ₀, fiber, abstol, reltol, maxiter, disable_warnings)

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
