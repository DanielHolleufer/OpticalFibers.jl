using Test
using OpticalFibers
using Integrals
using Bessels

@testset "Argument Checks" begin
    a = 0.05
    λ = 0.399
    SiO2 = Material(0.6961663, 0.4079426, 0.8974794, 0.0684043^2, 0.1162414^2, 9.896161^2)
    fiber = Fiber(a, λ, SiO2)

    @test_throws DomainError electric_guided_mode_cylindrical_base_components(-1.0, fiber)
    @test_throws DomainError electric_guided_mode_profile_cartesian_components(-1.0, 0.0, 1, 1, fiber, CircularPolarization())
    @test_throws DomainError electric_guided_mode_profile_cartesian_components(1.0, 0.0, 2, 1, fiber, CircularPolarization())
    @test_throws DomainError electric_guided_mode_profile_cartesian_components(1.0, 0.0, -2, 1, fiber, CircularPolarization())
    @test_throws DomainError electric_guided_mode_profile_cartesian_components(1.0, 0.0, 1, 2, fiber, CircularPolarization())
    @test_throws DomainError electric_guided_mode_profile_cartesian_components(1.0, 0.0, 1, -2, fiber, CircularPolarization())
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
        K1J1 = fiber.besselk1_over_besselj1
        A = fiber.normalization_constant
        
        if ρ < a
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
