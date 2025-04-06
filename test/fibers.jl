@testset "Fibers" begin
    function guided_mode_normalization_constant_integration(fiber::Fiber)
        parameters = (fiber,)
        domain = (0.0, Inf)
        problem = IntegralProblem(guided_mode_normalization_integrand, domain, parameters)
        solution = solve(problem, HCubatureJL(); reltol = 1e-10, abstol = 1e-10)
        return 1 / sqrt(2π * solution.u)
    end

    function guided_mode_normalization_integrand(ρ, parameters)
        fiber = parameters[1]
        a = fiber.radius
        n = fiber.refractive_index
        e_ρ, e_ϕ, e_z = OpticalFibers.electric_guided_mode_cylindrical_base_components(ρ, fiber)
        abs2e = abs2(e_ρ) + abs2(e_ϕ) + abs2(e_z)
        if ρ < a
            return ρ * n^2 * abs2e
        else
            return ρ * abs2e
        end
    end

    a = 0.05
    λ = 0.395
    SiO2 = Material(0.6961663, 0.4079426, 0.8974794, 0.0684043^2, 0.1162414^2, 9.896161^2)
    fiber = Fiber(a, λ, SiO2)
    normalization_constant_integration = guided_mode_normalization_constant_integration(fiber)
    @test fiber.normalization_constant ≈ normalization_constant_integration atol=1e-9

    a = 0.25
    λ = 0.852
    fiber = Fiber(a, λ, SiO2)
    normalization_constant_integration = guided_mode_normalization_constant_integration(fiber)
    @test fiber.normalization_constant ≈ normalization_constant_integration atol=1e-9
end
