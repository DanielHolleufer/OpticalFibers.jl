function gaussian_atomic_cloud(N, distribution, fiber)
    r = zeros(3, N)
    for i in 1:N
        while sqrt(r[1, i]^2 + r[2, i]^2) ≤ fiber.radius
            r[:, i] = rand(distribution)
        end
    end
    return r
end

function gaussian_beam_intensity(x::Real, y::Real, z::Real, waist::Real, power::Real, wavelength::Real)
    z_r = π * waist^2 / wavelength
    w = waist * sqrt(1 + (z / z_r)^2)
    intensity_max = 2 * power / (π * waist^2)
    intensity = intensity_max * (waist / w)^2 * exp(-2 * (x^2 + y^2) / w^2)
    return intensity
end

function tweezer_trap_intensity(x::Real, y::Real, z::Real, waist::Real, power::Real, wavelength::Real)
    tweezer_x = gaussian_beam_intensity(z, y, x, waist, power, wavelength)
    tweezer_y = gaussian_beam_intensity(x, z, y, waist, power, wavelength)
    return tweezer_x + tweezer_y
end

function tweezer_trap_potential(x::Real, y::Real, z::Real, waist::Real, power::Real, wavelength::Real, stark_shift::Real)
    return stark_shift * tweezer_trap_intensity(z, y, x, waist, power, wavelength)
end

function fiber_potential(x::Real, y::Real, z::Real, l::Integer, f::Integer, fiber::Fiber, polarization_basis::CircularPolarization, power::Real, stark_shift::Real)
    ρ = sqrt(x^2 + y^2)
    ϕ = atan(y, x)
    e_x, e_y, e_z = electric_guided_field_cartesian_components(ρ, ϕ, z, 0.0, l, f, fiber, polarization_basis, power)
    e_abs2 = abs2(e_x) + abs2(e_y) + abs2(e_z)
    return stark_shift * e_abs2
end

function fiber_potential(x::Real, y::Real, z::Real, f::Integer, fiber::Fiber, polarization_basis::LinearPolarization, power::Real, stark_shift::Real)
    ρ = sqrt(x^2 + y^2)
    ϕ = atan(y, x)
    e_x1, e_y1, e_z1 = electric_guided_field_cartesian_components(ρ, ϕ, z, 0.0, 1, f, fiber, polarization_basis, power)
    e_x2, e_y2, e_z2 = electric_guided_field_cartesian_components(ρ, ϕ, z, 0.0, -1, f, fiber, polarization_basis, power)
    e_x = 1 / sqrt(2) * (e_x1 + e_x2)
    e_y = 1 / sqrt(2) * (e_y1 + e_y2)
    e_z = 1 / sqrt(2) * (e_z1 + e_z2)
    e_abs2 = abs2(e_x) + abs2(e_y) + abs2(e_z)
    return stark_shift * e_abs2
end

function atomic_propability_distribution(x::Real, y::Real, z::Real, waist::Real, power_trap::Real, wavelength_trap::Real, stark_shift_trap::Real, l::Integer, f::Integer, fiber::Fiber, polarization_basis::PolarizationBasis, power_fiber::Real, stark_shift_fiber::Real, temperature::Real)
    V_trap = tweezer_trap_potential(x, y, z, waist, power_trap, wavelength_trap, stark_shift_trap)
    V_fiber = fiber_potential(x, y, z, l, f, fiber, polarization_basis, power_fiber, stark_shift_fiber)
    return exp(-(V_trap + V_fiber) / temperature)
end
