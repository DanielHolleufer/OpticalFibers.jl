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

    @test_throws DomainError electric_guided_mode_cylindrical_base_components(-1.0, fiber)

    linear_x = LinearPolarization(0.0)
    @test_throws DomainError GuidedMode(fiber, linear_x, 0)
    @test_throws DomainError GuidedMode(fiber, linear_x, 2)
    @test_throws MethodError GuidedMode(fiber, linear_x, 1.0)
    
    circular_clockwise = CircularPolarization(-1)
    @test_throws DomainError GuidedMode(fiber, circular_clockwise, 0)
    @test_throws DomainError GuidedMode(fiber, circular_clockwise, 2)
    @test_throws MethodError GuidedMode(fiber, circular_clockwise, 1.0)

    linear_mode = GuidedMode(fiber, linear_x, 1)
    circular_mode = GuidedMode(fiber, circular_clockwise, 1)
    @test typeof(linear_mode) != typeof(circular_mode)

    linear_field = GuidedField(linear_mode, 1.0)
    circular_field = GuidedField(circular_mode, 1.0)
    @test typeof(linear_field) != typeof(circular_field)
    @test typeof(linear_mode) != typeof(linear_field)

    @test direction(linear_field) == direction(linear_mode)
    @test polarization(linear_field) == polarization(linear_mode)
    @test frequency(linear_field) == frequency(linear_mode)
    @test propagation_constant(linear_field) == propagation_constant(linear_mode)
    @test propagation_constant_derivative(linear_field) == propagation_constant_derivative(linear_mode)
end

@testset "Transformation Between Linearly and Circularly Polarized Modes" begin
    function circular_linear_transformation(fiber)
        test_resolution = 513
        xs = LinRange(-25.0, 25.0, test_resolution)
        ys = LinRange(-25.0, 25.0, test_resolution)

        linearly_polarized_xx_1 = zeros(ComplexF64, length(xs), length(ys))
        linearly_polarized_xy_1 = zeros(ComplexF64, length(xs), length(ys))
        linearly_polarized_xz_1 = zeros(ComplexF64, length(xs), length(ys))
        linearly_polarized_xx_2 = zeros(ComplexF64, length(xs), length(ys))
        linearly_polarized_xy_2 = zeros(ComplexF64, length(xs), length(ys))
        linearly_polarized_xz_2 = zeros(ComplexF64, length(xs), length(ys))
        linearly_polarized_yx_1 = zeros(ComplexF64, length(xs), length(ys))
        linearly_polarized_yy_1 = zeros(ComplexF64, length(xs), length(ys))
        linearly_polarized_yz_1 = zeros(ComplexF64, length(xs), length(ys))
        linearly_polarized_yx_2 = zeros(ComplexF64, length(xs), length(ys))
        linearly_polarized_yy_2 = zeros(ComplexF64, length(xs), length(ys))
        linearly_polarized_yz_2 = zeros(ComplexF64, length(xs), length(ys))

        linear_mode_x = GuidedMode(fiber, LinearPolarization('x'), 1)
        linear_mode_y = GuidedMode(fiber, LinearPolarization('y'), 1)
        circular_mode_p = GuidedMode(fiber, CircularPolarization(1), 1)
        circular_mode_m = GuidedMode(fiber, CircularPolarization(-1), 1)

        for (j, y) in enumerate(ys)
            for (i, x) in enumerate(xs)
                ρ = sqrt(x^2 + y^2)
                ϕ = atan(y, x)
                E_lin_x_x, E_lin_x_y, E_lin_x_z = electric_guided_mode_profile_cartesian_components(ρ, ϕ, linear_mode_x)
                E_lin_y_x, E_lin_y_y, E_lin_y_z = electric_guided_mode_profile_cartesian_components(ρ, ϕ, linear_mode_y)
                E_circ_p_x, E_circ_p_y, E_circ_p_z = electric_guided_mode_profile_cartesian_components(ρ, ϕ, circular_mode_p)
                E_circ_m_x, E_circ_m_y, E_circ_m_z = electric_guided_mode_profile_cartesian_components(ρ, ϕ, circular_mode_m)

                linearly_polarized_xx_1[i, j] = E_lin_x_x
                linearly_polarized_xy_1[i, j] = E_lin_x_y
                linearly_polarized_xz_1[i, j] = E_lin_x_z
                linearly_polarized_yx_1[i, j] = E_lin_y_x
                linearly_polarized_yy_1[i, j] = E_lin_y_y
                linearly_polarized_yz_1[i, j] = E_lin_y_z
                linearly_polarized_xx_2[i, j] = 1 / sqrt(2) * (E_circ_p_x + E_circ_m_x)
                linearly_polarized_xy_2[i, j] = 1 / sqrt(2) * (E_circ_p_y + E_circ_m_y)
                linearly_polarized_xz_2[i, j] = 1 / sqrt(2) * (E_circ_p_z + E_circ_m_z)
                linearly_polarized_yx_2[i, j] = -im / sqrt(2) * (E_circ_p_x - E_circ_m_x)
                linearly_polarized_yy_2[i, j] = -im / sqrt(2) * (E_circ_p_y - E_circ_m_y)
                linearly_polarized_yz_2[i, j] = -im / sqrt(2) * (E_circ_p_z - E_circ_m_z)
            end
        end

        return linearly_polarized_xx_1, linearly_polarized_xy_1, linearly_polarized_xz_1,
            linearly_polarized_yx_1, linearly_polarized_yy_1, linearly_polarized_yz_1,
            linearly_polarized_xx_2, linearly_polarized_xy_2, linearly_polarized_xz_2,
            linearly_polarized_yx_2, linearly_polarized_yy_2, linearly_polarized_yz_2
    end

    a = 0.05
    λ = 0.399
    SiO2 = Material(0.6961663, 0.4079426, 0.8974794, 0.0684043^2, 0.1162414^2, 9.896161^2)
    fiber = Fiber(a, λ, SiO2)
    lin_xx_1, lin_xy_1, lin_xz_1, lin_yx_1, lin_yy_1, lin_yz_1, lin_xx_2, lin_xy_2, lin_xz_2, lin_yx_2, lin_yy_2, lin_yz_2 = circular_linear_transformation(fiber)
    @test lin_xx_1 ≈ lin_xx_2 atol = 1e-12
    @test lin_xy_1 ≈ lin_xy_2 atol = 1e-12
    @test lin_xz_1 ≈ lin_xz_2 atol = 1e-12
    @test lin_yx_1 ≈ lin_yx_2 atol = 1e-12
    @test lin_yy_1 ≈ lin_yy_2 atol = 1e-12
    @test lin_yz_1 ≈ lin_yz_2 atol = 1e-12

    a = 0.2
    λ = 0.852
    fiber = Fiber(a, λ, SiO2)
    lin_xx_1, lin_xy_1, lin_xz_1, lin_yx_1, lin_yy_1, lin_yz_1, lin_xx_2, lin_xy_2, lin_xz_2, lin_yx_2, lin_yy_2, lin_yz_2 = circular_linear_transformation(fiber)
    @test lin_xx_1 ≈ lin_xx_2 atol = 1e-12
    @test lin_xy_1 ≈ lin_xy_2 atol = 1e-12
    @test lin_xz_1 ≈ lin_xz_2 atol = 1e-12
    @test lin_yx_1 ≈ lin_yx_2 atol = 1e-12
    @test lin_yy_1 ≈ lin_yy_2 atol = 1e-12
    @test lin_yz_1 ≈ lin_yz_2 atol = 1e-12
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
        e_ρ, e_ϕ, e_z = electric_guided_mode_cylindrical_base_components(ρ, fiber)
        de_z = longitudinal_base_component_derivative(ρ::Real, fiber::Fiber)
        return 2π * dβ / ω * ρ * real(β * e_ϕ^2 - β * e_ρ^2 - e_ϕ * e_z / ρ - im * e_ρ * de_z)
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
