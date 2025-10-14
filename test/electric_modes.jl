using Test
using OpticalFibers
using Integrals
using Bessels

@testset "Polarization" begin
    linear_x = LinearPolarization(0.0)
    @test linear_x isa LinearPolarization
    @test !(linear_x isa CircularPolarization)
    @test linear_x == LinearPolarization(0)
    @test linear_x == LinearPolarization('x')
    @test linear_x == LinearPolarization('X')
    @test linear_x == LinearPolarization("x")
    @test linear_x == LinearPolarization("X")
    @test linear_x == LinearPolarization(:x)
    @test linear_x == LinearPolarization(:X)

    linear_y = LinearPolarization(π / 2)
    @test linear_y == LinearPolarization('y')
    @test linear_y == LinearPolarization('Y')
    @test linear_y == LinearPolarization("y")
    @test linear_y == LinearPolarization("Y")
    @test linear_y == LinearPolarization(:y)
    @test linear_y == LinearPolarization(:Y)

    @test LinearPolarization(0.0) != LinearPolarization(π / 2)

    @test_throws ErrorException LinearPolarization('z')
    @test_throws ArgumentError LinearPolarization("xx")
    @test_throws ArgumentError LinearPolarization(:yy)

    circular_counterclockwise = CircularPolarization(1)
    @test circular_counterclockwise isa CircularPolarization
    @test !(circular_counterclockwise isa LinearPolarization)
    
    circular_clockwise = CircularPolarization(-1)
    @test circular_counterclockwise != circular_clockwise 
    
    @test_throws MethodError CircularPolarization(1.0)
    @test_throws DomainError CircularPolarization(2)
end

@testset "Guided Modes and Fields" begin
    a = 0.05
    λ = 0.399
    SiO2 = Material(0.6961663, 0.4079426, 0.8974794, 0.0684043^2, 0.1162414^2, 9.896161^2)
    fiber = Fiber(a, λ, SiO2)

    @test_throws DomainError electric_guided_mode_base(-1.0, fiber)

    HLP = LinearPolarization(0.0)
    @test_throws DomainError GuidedMode(fiber, HLP, 0)
    @test_throws DomainError GuidedMode(fiber, HLP, 2)
    @test_throws MethodError GuidedMode(fiber, HLP, 1.0)
    
    LHCP = CircularPolarization(-1)
    @test_throws DomainError GuidedMode(fiber, LHCP, 0)
    @test_throws DomainError GuidedMode(fiber, LHCP, 2)
    @test_throws MethodError GuidedMode(fiber, LHCP, 1.0)

    HLP_mode = GuidedMode(fiber, HLP, 1)
    LHCP_mode = GuidedMode(fiber, LHCP, 1)
    @test typeof(HLP_mode) != typeof(LHCP_mode)

    HLP_field = GuidedField(HLP_mode, 1.0)
    LHCP_field = GuidedField(LHCP_mode, 1.0)
    @test typeof(HLP_field) != typeof(LHCP_field)
    @test typeof(HLP_mode) != typeof(HLP_field)

    @test direction(HLP_mode) == direction(HLP_mode)
    @test polarization(HLP_field) == polarization(HLP_mode)
    @test frequency(HLP_field) == frequency(HLP_mode)
    @test propagation_constant(HLP_field) == propagation_constant(HLP_mode)
    @test propagation_constant_derivative(HLP_field) == propagation_constant_derivative(HLP_mode)
end

@testset "Transformation Between Linearly and Circularly Polarized Modes" begin
    function mode_components_transverse_plane(xs, ys, mode)
        Ex = Matrix{ComplexF64}(undef, length(xs), length(ys))
        Ey = Matrix{ComplexF64}(undef, length(xs), length(ys))
        Ez = Matrix{ComplexF64}(undef, length(xs), length(ys))

        for (j, y) in enumerate(ys)
            for (i, x) in enumerate(xs)
                ρ = sqrt(x^2 + y^2)
                ϕ = atan(y, x)
                E_x, E_y, E_z = electric_guided_mode_profile_cartesian(ρ, ϕ, mode)
                Ex[i, j] = E_x
                Ey[i, j] = E_y
                Ez[i, j] = E_z
            end
        end

        return Ex, Ey, Ez
    end

    resolution = 251
    xs = LinRange(-25.0, 25.0, resolution)
    ys = LinRange(-25.0, 25.0, resolution)

    a = 0.05
    λ = 0.399
    SiO2 = Material(0.6961663, 0.4079426, 0.8974794, 0.0684043^2, 0.1162414^2, 9.896161^2)
    fiber = Fiber(a, λ, SiO2)
    forward = 1
    HLP = GuidedMode(fiber, LinearPolarization('x'), forward)
    VLP = GuidedMode(fiber, LinearPolarization('y'), forward)
    RHCP = GuidedMode(fiber, CircularPolarization(1), forward)
    LHCP = GuidedMode(fiber, CircularPolarization(-1), forward)

    Ex_HLP, Ey_HLP, Ez_HLP = mode_components_transverse_plane(xs, ys, HLP)
    Ex_VLP, Ey_VLP, Ez_VLP = mode_components_transverse_plane(xs, ys, VLP)
    Ex_RHCP, Ey_RHCP, Ez_RHCP = mode_components_transverse_plane(xs, ys, RHCP)
    Ex_LHCP, Ey_LHCP, Ez_LHCP = mode_components_transverse_plane(xs, ys, LHCP)

    @test Ex_HLP ≈ 1 / sqrt(2) * (Ex_RHCP + Ex_LHCP) atol=1e-12
    @test Ey_HLP ≈ 1 / sqrt(2) * (Ey_RHCP + Ey_LHCP) atol=1e-12
    @test Ez_HLP ≈ 1 / sqrt(2) * (Ez_RHCP + Ez_LHCP) atol=1e-12
    @test Ex_VLP ≈ -im / sqrt(2) * (Ex_RHCP - Ex_LHCP) atol=1e-12
    @test Ey_VLP ≈ -im / sqrt(2) * (Ey_RHCP - Ey_LHCP) atol=1e-12
    @test Ez_VLP ≈ -im / sqrt(2) * (Ez_RHCP - Ez_LHCP) atol=1e-12

    a = 0.25
    λ = 0.852
    SiO2 = Material(0.6961663, 0.4079426, 0.8974794, 0.0684043^2, 0.1162414^2, 9.896161^2)
    fiber = Fiber(a, λ, SiO2)
    backward = -1
    HLP = GuidedMode(fiber, LinearPolarization('x'), backward)
    VLP = GuidedMode(fiber, LinearPolarization('y'), backward)
    RHCP = GuidedMode(fiber, CircularPolarization(1), backward)
    LHCP = GuidedMode(fiber, CircularPolarization(-1), backward)

    Ex_HLP, Ey_HLP, Ez_HLP = mode_components_transverse_plane(xs, ys, HLP)
    Ex_VLP, Ey_VLP, Ez_VLP = mode_components_transverse_plane(xs, ys, VLP)
    Ex_RHCP, Ey_RHCP, Ez_RHCP = mode_components_transverse_plane(xs, ys, RHCP)
    Ex_LHCP, Ey_LHCP, Ez_LHCP = mode_components_transverse_plane(xs, ys, LHCP)

    @test Ex_HLP ≈ 1 / sqrt(2) * (Ex_RHCP + Ex_LHCP) atol=1e-12
    @test Ey_HLP ≈ 1 / sqrt(2) * (Ey_RHCP + Ey_LHCP) atol=1e-12
    @test Ez_HLP ≈ 1 / sqrt(2) * (Ez_RHCP + Ez_LHCP) atol=1e-12
    @test Ex_VLP ≈ -im / sqrt(2) * (Ex_RHCP - Ex_LHCP) atol=1e-12
    @test Ey_VLP ≈ -im / sqrt(2) * (Ey_RHCP - Ey_LHCP) atol=1e-12
    @test Ez_VLP ≈ -im / sqrt(2) * (Ez_RHCP - Ez_LHCP) atol=1e-12
end

@testset "Propagating Power" begin
    function propagating_power(fiber::Fiber)
        domain = (0.0, Inf)
        p = (fiber,)
        problem = IntegralProblem(propagating_power_integrand, domain, p)
        solution = solve(problem, HCubatureJL(); reltol = 1e-15, abstol = 1e-15)
        return solution.u
    end

    function propagating_power_integrand(u, p)
        ρ = u[1]
        fiber = p[1]
        ω = fiber.frequency
        β = fiber.propagation_constant
        dβ = propagation_constant_derivative(fiber)
        e_ρ, e_ϕ, e_z = electric_guided_mode_base(ρ, fiber)
        de_z = longitudinal_base_component_derivative(ρ::Real, fiber::Fiber)
        P = 2π * dβ / ω * ρ * real(β * e_ϕ^2 - β * e_ρ^2 - e_ϕ * e_z / ρ - im * e_ρ * de_z)
        return P
    end

    function longitudinal_base_component_derivative(ρ::Real, fiber::Fiber)
        a = fiber.radius
        β = fiber.propagation_constant
        h = fiber.internal_parameter
        q = fiber.external_parameter
        A = fiber.normalization_constant
        
        if ρ < a
            K1J1 = besselk1(q * a) / besselj1(h * a)
            de_z = A * h * q / β * K1J1 * (besselj0(h * ρ) - besselj(2, h * ρ))
        else
            de_z = -A * q^2 / β * (besselk(0, q * ρ) + besselk(2, q * ρ))
        end

        return de_z
    end

    a = 0.05
    λ = 0.399
    SiO2 = Material(0.6961663, 0.4079426, 0.8974794, 0.0684043^2, 0.1162414^2, 9.896161^2)
    fiber = Fiber(a, λ, SiO2)
    @test propagating_power(fiber) ≈ 1.0 atol=1e-5

    λ = 0.395
    fiber = Fiber(a, λ, SiO2)
    @test propagating_power(fiber) ≈ 1.0 atol=1e-5

    a = 0.25
    λ = 0.852
    fiber = Fiber(a, λ, SiO2)
    @test propagating_power(fiber) ≈ 1.0 atol=1e-5
end

@testset "Auxiliary Coefficients" begin
    gauss_legendre_pairs = OpticalFibers.gauss_legendre_pairs
    BesselHankelSurfaceEvaluations = OpticalFibers.BesselHankelSurfaceEvaluations
    bessel_hankel_surface_evaluations = OpticalFibers.bessel_hankel_surface_evaluations
    RadiationAuxiliaryCoefficients = OpticalFibers.RadiationAuxiliaryCoefficients
    radiation_auxiliary_coefficients = OpticalFibers.radiation_auxiliary_coefficients

    V_coefficient(coefficients::RadiationAuxiliaryCoefficients) = coefficients.V
    M_coefficient(coefficients::RadiationAuxiliaryCoefficients) = coefficients.M
    L_coefficient(coefficients::RadiationAuxiliaryCoefficients) = coefficients.L
    η_coefficient(coefficients::RadiationAuxiliaryCoefficients) = coefficients.η

    besselj_derivative(m, x) = 0.5 * (besselj(m - 1, x) - besselj(m + 1, x))
    hankelh1_derivative(m, x) = 0.5 * (hankelh1(m - 1, x) - hankelh1(m + 1, x))

    a = 0.05
    λ = 0.399
    ω = 2π / λ
    SiO2 = Material(0.6961663, 0.4079426, 0.8974794, 0.0684043^2, 0.1162414^2, 9.896161^2)
    fiber = Fiber(a, λ, SiO2)
    n = refractive_index(fiber)

    m_max = 50
    resolution = 1000
    βs = ω * cos.(gauss_legendre_pairs(resolution))
    hs = sqrt.(n^2 * ω^2 .- βs.^2)
    qs = sqrt.(ω^2 .- βs.^2)

    maximum(abs.(hs[1:500] - reverse(hs[501:end])))
    maximum(abs.(qs[1:500] - reverse(qs[501:end])))

    evaluations = bessel_hankel_surface_evaluations(m_max, hs, qs, a, resolution)
    auxiliary = radiation_auxiliary_coefficients(ω, βs, m_max, hs, qs, a, n, evaluations)
    m_idx = (1, 5, 6, 31, 32, 49, m_max)
    β_idx = (1, 2, 3, 99, 100, 101, 499, 500, 501, 749, 750, 751, 999, resolution)
    for j in β_idx, i in m_idx,
        m = i - 1
        β = βs[j]
        h = hs[j]
        q = qs[j]

        J = besselj(m, h * a)
        dJ = besselj_derivative(m, h * a)
        H1 = hankelh1(m, q * a)
        dH1 = hankelh1_derivative(m, q * a)
        evals = BesselHankelSurfaceEvaluations(J, dJ, H1, dH1)
        @test evals == evaluations[i, j]

        aux = radiation_auxiliary_coefficients(ω, β, m, h, q, a, n, evals)
        @test aux == auxiliary[i, j]
    end

    a = 0.25
    λ = 0.852
    ω = 2π / λ
    fiber = Fiber(a, λ, SiO2)
    n = refractive_index(fiber)

    m_max = 35
    resolution = 513
    βs = ω * cos.(gauss_legendre_pairs(resolution))
    hs = sqrt.(n^2 * ω^2 .- βs.^2)
    qs = sqrt.(ω^2 .- βs.^2)

    evaluations = bessel_hankel_surface_evaluations(m_max, hs, qs, a, resolution)
    auxiliary = radiation_auxiliary_coefficients(ω, βs, m_max, hs, qs, a, n, evaluations)
    m_idx = (1, 2, 9, 10, 34, m_max)
    β_idx = (1, 2, 5, 199, 200, 201, 499, 500, 501, resolution)
    for j in β_idx, i in m_idx,
        m = i - 1
        β = βs[j]
        h = hs[j]
        q = qs[j]

        J = besselj(m, h * a)
        dJ = besselj_derivative(m, h * a)
        H1 = hankelh1(m, q * a)
        dH1 = hankelh1_derivative(m, q * a)
        evals = BesselHankelSurfaceEvaluations(J, dJ, H1, dH1)
        @test evals == evaluations[i, j]

        aux = radiation_auxiliary_coefficients(ω, β, m, h, q, a, n, evals)
        @test aux == auxiliary[i, j]
    end
end

@testset "Boundary Coefficients" begin
    bessel_hankel_surface_evaluations = OpticalFibers.bessel_hankel_surface_evaluations
    RadiationAuxiliaryCoefficients = OpticalFibers.RadiationAuxiliaryCoefficients
    radiation_auxiliary_coefficients = OpticalFibers.radiation_auxiliary_coefficients
    RadiationBoundaryCoefficients = OpticalFibers.RadiationBoundaryCoefficients
    radiation_boundary_coefficients_arbitrary_dipole = OpticalFibers.radiation_boundary_coefficients_arbitrary_dipole
    gauss_legendre_pairs_positive = OpticalFibers.gauss_legendre_pairs_positive

    a = 0.05
    λ = 0.399
    ω = 2π / λ
    SiO2 = Material(0.6961663, 0.4079426, 0.8974794, 0.0684043^2, 0.1162414^2, 9.896161^2)
    fiber = Fiber(a, λ, SiO2)
    n = refractive_index(fiber)

    m_max = 20
    resolution = 500
    βs = ω * reverse(cos.(gauss_legendre_pairs_positive(resolution)))
    hs = sqrt.(n^2 * ω^2 .- βs.^2)
    qs = sqrt.(ω^2 .- βs.^2)
    βs_full = [-reverse(βs); βs]

    A_coefficient(coefficients::RadiationBoundaryCoefficients) = coefficients.A
    B_coefficient(coefficients::RadiationBoundaryCoefficients) = coefficients.B
    C_coefficient(coefficients::RadiationBoundaryCoefficients) = coefficients.C
    D_coefficient(coefficients::RadiationBoundaryCoefficients) = coefficients.D

    evaluations = bessel_hankel_surface_evaluations(m_max, hs, qs, a, resolution)
    auxiliary = radiation_auxiliary_coefficients(ω, βs, m_max, hs, qs, a, n, evaluations)
    boundary = radiation_boundary_coefficients_arbitrary_dipole(ω, m_max, βs, qs, a, auxiliary)

    A_coefficients1 = A_coefficient.(boundary[1:m_max + 1, :, 1])
    A_coefficients2 = A_coefficient.(reverse(boundary[m_max + 1:end, :, 2], dims=1))
    @test A_coefficients1 == A_coefficients2

    B_coefficients1 = B_coefficient.(boundary[:, :, 1])
    B_coefficients2 = B_coefficient.(boundary[:, :, 2])
    @test B_coefficients1 == -B_coefficients2

    C_coefficients1 = C_coefficient.(boundary[1:m_max + 1, :, 1])
    C_coefficients2 = C_coefficient.(reverse(boundary[m_max + 1:end, :, 2], dims=1))
    @test C_coefficients1 == C_coefficients2

    C_coefficients3 = C_coefficient.(boundary[1:m_max + 1, :, 2])
    C_coefficients4 = C_coefficient.(reverse(boundary[m_max + 1:end, :, 1], dims=1))
    @test C_coefficients3 == C_coefficients4
end

@testset "Radiation Mode Symmetries" begin
    # We test the following symmetries given in eqs. (B8), (B9), and (B10) from
    # 10.1103/PhysRevA.95.023838.
    a = 0.05
    λ = 0.399
    ω = 2π / λ
    SiO2 = Material(0.6961663, 0.4079426, 0.8974794, 0.0684043^2, 0.1162414^2, 9.896161^2)
    fiber = Fiber(a, λ, SiO2)
    ρ = 0.1
    β = ω / 2
    m = 0
    eρ1, eϕ1, ez1 = electric_radiation_mode_base(ρ, ω, β, m, 1, fiber)
    eρ2, eϕ2, ez2 = electric_radiation_mode_base(ρ, ω, -β, m, -1, fiber)
    eρ3, eϕ3, ez3 = electric_radiation_mode_base(ρ, ω, β, -m, -1, fiber)
    @test eρ1 == -eρ2
    @test eϕ1 == -eϕ2
    @test ez1 == ez2
    @test eρ1 == (-1)^m * eρ3
    @test eϕ1 == (-1)^(m + 1) * eϕ3
    @test ez1 == (-1)^m * ez3
    @test eρ1' == -eρ1
    @test eρ2' == -eρ2
    @test eρ3' == -eρ3
    @test isreal(eϕ1)
    @test isreal(eϕ2)
    @test isreal(eϕ3)
    @test isreal(ez1)
    @test isreal(ez2)
    @test isreal(ez3)

    ρ = 0.21
    β = sqrt(2) / 3 * ω
    m = 1
    eρ1, eϕ1, ez1 = electric_radiation_mode_base(ρ, ω, β, m, 1, fiber)
    eρ2, eϕ2, ez2 = electric_radiation_mode_base(ρ, ω, -β, m, -1, fiber)
    eρ3, eϕ3, ez3 = electric_radiation_mode_base(ρ, ω, β, -m, -1, fiber)
    @test eρ1 == -eρ2
    @test eϕ1 == -eϕ2
    @test ez1 == ez2
    @test eρ1 == (-1)^m * eρ3
    @test eϕ1 == (-1)^(m + 1) * eϕ3
    @test ez1 == (-1)^m * ez3
    @test eρ1' == -eρ1
    @test eρ2' == -eρ2
    @test eρ3' == -eρ3
    @test isreal(eϕ1)
    @test isreal(eϕ2)
    @test isreal(eϕ3)
    @test isreal(ez1)
    @test isreal(ez2)
    @test isreal(ez3)

    a = 0.25
    λ = 0.852
    ω = 2π / λ
    SiO2 = Material(0.6961663, 0.4079426, 0.8974794, 0.0684043^2, 0.1162414^2, 9.896161^2)
    fiber = Fiber(a, λ, SiO2)
    ρ = 0.3
    β = -ω / 10
    m = 2
    eρ1, eϕ1, ez1 = electric_radiation_mode_base(ρ, ω, β, m, 1, fiber)
    eρ2, eϕ2, ez2 = electric_radiation_mode_base(ρ, ω, -β, m, -1, fiber)
    eρ3, eϕ3, ez3 = electric_radiation_mode_base(ρ, ω, β, -m, -1, fiber)
    @test eρ1 == -eρ2
    @test eϕ1 == -eϕ2
    @test ez1 == ez2
    @test eρ1 == (-1)^m * eρ3
    @test eϕ1 == (-1)^(m + 1) * eϕ3
    @test ez1 == (-1)^m * ez3
    @test eρ1' == -eρ1
    @test eρ2' == -eρ2
    @test eρ3' == -eρ3
    @test isreal(eϕ1)
    @test isreal(eϕ2)
    @test isreal(eϕ3)
    @test isreal(ez1)
    @test isreal(ez2)
    @test isreal(ez3)

    ρ = 1.01
    β = 0.99 * ω
    m = -21
    eρ1, eϕ1, ez1 = electric_radiation_mode_base(ρ, ω, β, m, 1, fiber)
    eρ2, eϕ2, ez2 = electric_radiation_mode_base(ρ, ω, -β, m, -1, fiber)
    eρ3, eϕ3, ez3 = electric_radiation_mode_base(ρ, ω, β, -m, -1, fiber)
    @test eρ1 == -eρ2
    @test eϕ1 == -eϕ2
    @test ez1 == ez2
    @test eρ1 == (-1)^m * eρ3
    @test eϕ1 == (-1)^(m + 1) * eϕ3
    @test ez1 == (-1)^m * ez3
    @test eρ1' == -eρ1
    @test eρ2' == -eρ2
    @test eρ3' == -eρ3
    @test isreal(eϕ1)
    @test isreal(eϕ2)
    @test isreal(eϕ3)
    @test isreal(ez1)
    @test isreal(ez2)
    @test isreal(ez3)
end
