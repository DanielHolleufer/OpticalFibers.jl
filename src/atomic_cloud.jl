function gaussian_atomic_cloud(N, distribution, fiber)
    r = zeros(3, N)
    for i in 1:N
        while sqrt(r[1, i]^2 + r[2, i]^2) ≤ fiber.radius
            r[:, i] = rand(distribution)
        end
    end
    return r
end

"""
    gaussian_beam_intensity(x::Real, y::Real, z::Real, waist::Real, power::Real, wavelength::Real)

Compute the intensity at position (`x`, `y`, `z`) of a gaussian beam parallel to the z-axis
with parameters `waist`, `power`, and `wavelength`.
"""
function gaussian_beam_intensity(x::Real, y::Real, z::Real, waist::Real, power::Real, wavelength::Real)
    z_r = π * waist^2 / wavelength
    w = waist * sqrt(1 + (z / z_r)^2)
    intensity_max = 2 * power / (π * waist^2)
    intensity = intensity_max * (waist / w)^2 * exp(-2 * (x^2 + y^2) / w^2)
    return intensity
end

"""
    tweezer_trap_intensity(x::Real, y::Real, z::Real, waist::Real, power::Real, wavelength::Real)

Compute the intensity at position (`x`, `y`, `z`) of a dipole tweezer trap consisting of two
beam parallel to the gaussian beams, one along the x-axis and one along the y-axis, where
both beams have parameters `waist`, `power`, and `wavelength`.
"""
function tweezer_trap_intensity(x::Real, y::Real, z::Real, waist::Real, power::Real, wavelength::Real)
    tweezer_x = gaussian_beam_intensity(z, y, x, waist, power, wavelength)
    tweezer_y = gaussian_beam_intensity(x, z, y, waist, power, wavelength)
    return tweezer_x + tweezer_y
end

"""
    tweezer_trap_potential(x::Real, y::Real, z::Real, waist::Real, power::Real, wavelength::Real, stark_shift::Real)

Compute the potential at position (`x`, `y`, `z`) of a dipole tweezer trap consisting of two
beam parallel to the gaussian beams, one along the x-axis and one along the y-axis, where
both beams have parameters `waist`, `power`, and `wavelength`, and the stark_shift per unit
intensity is `stark_shift`.
"""
function tweezer_trap_potential(x::Real,y::Real, z::Real, waist::Real, power::Real, wavelength::Real, stark_shift::Real)
    return stark_shift * tweezer_trap_intensity(x, y, z, waist, power, wavelength)
end

"""
    fiber_potential(x::Real, y::Real, z::Real, l::Integer, f::Integer, fiber::Fiber, polarization_basis::CircularPolarization, power::Real, stark_shift::Real)

Compute the potential at position (`x`, `y`, `z`) of a circularly polarized fiber mode along
the z-axis with polarization index `l`, direction of propagation index `f`, `power`, and the
stark_shift per unit intensity is `stark_shift`.
"""
function fiber_potential(x::Real, y::Real, z::Real, l::Integer, f::Integer, fiber::Fiber, polarization_basis::CircularPolarization, power::Real, stark_shift::Real)
    ρ = sqrt(x^2 + y^2)
    ϕ = atan(y, x)
    e_x, e_y, e_z = electric_guided_field_cartesian_components(ρ, ϕ, z, 0.0, l, f, fiber, polarization_basis, power)
    e_abs2 = abs2(e_x) + abs2(e_y) + abs2(e_z)
    return stark_shift * e_abs2
end

"""
    full_potential(x::Real, y::Real, z::Real, waist::Real, power_trap::Real, wavelength::Real, stark_shift_trap::Real, l::Integer, f::Integer, fiber::Fiber, polarization_basis::CircularPolarization, power_fiber::Real, stark_shift_fiber::Real)

Compute the full potential at position (`x`, `y`, `z`) of a circularly polarized fiber mode
along the z-axis and a dipole tweezer trap consisting of two beam parallel to the gaussian
beams, one along the x-axis and one along the y-axis.
"""
function full_potential(x::Real, y::Real, z::Real, waist::Real, power_trap::Real, wavelength::Real, stark_shift_trap::Real, l::Integer, f::Integer, fiber::Fiber, polarization_basis::CircularPolarization, power_fiber::Real, stark_shift_fiber::Real)
    V_trap = tweezer_trap_potential(x, y, z, waist, power_trap, wavelength, stark_shift_trap)
    V_fiber = fiber_potential(x, y, z, l, f, fiber, polarization_basis, power_fiber, stark_shift_fiber)
    V_total = V_trap + V_fiber
    if V_total > 0
        return 0.0
    else
        return V_total
    end
end

function atomic_propability_distribution(x::Real, y::Real, z::Real, waist::Real, power_trap::Real, wavelength_trap::Real, stark_shift_trap::Real, l::Integer, f::Integer, fiber::Fiber, polarization_basis::PolarizationBasis, power_fiber::Real, stark_shift_fiber::Real, temperature::Real)
    ρ = sqrt(x^2 + y^2)
    if ρ < radius(fiber)
        return 0.0
    end
    
    V_total = full_potential(x, y, z, waist, power_trap, wavelength_trap, stark_shift_trap, l, f, fiber, polarization_basis, power_fiber, stark_shift_fiber)

    if -V_total / temperature < 1e-10
        return 0.0
    end
    P, _ = gamma_inc(3 / 2, -V_total / temperature)
    propability = 2 / sqrt(π) * P * exp(-V_total / temperature)
    #propability = (sqrt(π) / 2 - gamma(3 / 2, -V_total / temperature)) * exp(-V_total / temperature)

    if propability < 0.0
        return 0.0
    else
        return propability
    end
end

function atomic_propability_normalization_constant(waist, power_trap, wavelength_trap, stark_shift_trap, l, f, fiber, polarization_basis, power_fiber, stark_shift_fiber, temperature)
    #domain = (-Inf, Inf)
    #domain = ([-Inf, -Inf], [Inf, Inf])
    domain = ([-Inf, -Inf, -Inf], [Inf, Inf, Inf])
    #domain = ([-700.0, -700.0], [700.0, 700.0])
    p = waist, power_trap, wavelength_trap, stark_shift_trap, l, f, fiber, polarization_basis, power_fiber, stark_shift_fiber, temperature
    prob = IntegralProblem(atomic_propability_normalization_constant_integrand, domain, p)
    sol = solve(prob, HCubatureJL())
    return sol.u
end

function atomic_propability_normalization_constant_integrand(u, p)
    waist, power_trap, wavelength_trap, stark_shift_trap, l, f, fiber, polarization_basis, power_fiber, stark_shift_fiber, temperature = p
    return atomic_propability_distribution(u[1], u[2], u[3], waist, power_trap, wavelength_trap, stark_shift_trap, l, f, fiber, polarization_basis, power_fiber, stark_shift_fiber, temperature)
end

function lower_incomplete_gamma_three_halfs(x::Real)
    coefficients = (0.6666666666666666, 0.4, 0.14285714285714285, 0.037037037037037035, 0.007575757575757576, 0.0012820512820512823, 0.00018518518518518523, 2.3342670401493936e-5, 2.610693400167085e-6, 2.6245065927605617e-7, 2.3962886281726868e-8, 2.004168670835338e-9, 1.5464264435457854e-10, 1.1075202646083875e-11, 7.400481030793372e-13, 4.6346446859514054e-14, 2.7311299042213638e-15, 1.5197066239705522e-16, 8.009849727480118e-18, 4.0100659739630883e-19, 1.9117756387498445e-20, 8.699084917062786e-22, 3.785868677638543e-23, 1.5788449676043614e-24, 6.320539494494584e-26, 2.4328114280696132e-27, 9.01671368445381e-29, 3.2223473206300165e-30, 1.111826860023674e-31, 3.708184552933678e-33, 1.196821469465367e-34, 3.7419232047800556e-36, 1.1344450014491774e-37, 3.338068295875928e-39, 9.541288832453976e-41, 2.6513953311320248e-42, 7.168587376764363e-44, 1.8871325140657327e-45, 4.840413177317169e-47, 1.2104863596329736e-48, 2.953295034044304e-50, 7.033672534452718e-52, 1.636185455469297e-53, 3.719574983690327e-55, 8.267786552158819e-57, 1.797774375738238e-58, 3.8259271611820626e-60, 7.972429925691948e-62, 1.6273689031820686e-63, 3.2553954620130286e-65, 6.384367799287687e-67, 1.227992312468033e-68, 2.3173830483311193e-70, 4.2921929404782716e-72, 7.80528913099986e-74, 1.3940258946757591e-75, 2.446039225129826e-77, 4.2179413838645973e-79, 7.150088987426222e-81, 1.1918484234538737e-82, 1.9541146238200367e-84, 3.1522111308834694e-86, 5.0041451785678645e-88, 7.819938940299234e-90, 1.2032110249267666e-91, 1.823257886239519e-93, 2.7215858459018636e-95, 4.002768157716e-97, 5.801727016579475e-99, 8.289033357020733e-101, 1.1675861172227007e-102, 1.6218049029902495e-104, 2.221860458556181e-106, 3.0027901756712206e-108, 4.004078541032861e-110, 5.268983526762196e-112, 6.843416634928829e-114, 8.774336822019757e-116, 1.1107651032552024e-117, 1.388565543026788e-119, 1.7144099112523994e-121, 2.0909002284634575e-123, 2.5193408623957243e-125, 2.99942912967909e-127, 3.528985818127027e-129, 4.1037509343741696e-131, 4.717268515925125e-133, 5.3608805135846284e-135, 6.023843644644992e-137, 6.6935750970976074e-139, 7.356023634333133e-141, 7.99615280714561e-143, 8.598513539420703e-145, 9.147875245330099e-147, 9.629878697601586e-149, 1.0031670745797126e-150, 1.0342481057365627e-152, 1.0554104381109934e-154, 1.0661258143157917e-156, 1.0661793911696193e-158, 1.0556751607147463e-160)
    return x^(3 / 2) * evalpoly(-x, coefficients)
end

function coeffs(a)
    b = a + 1
    c = zeros(101)
    c[1] = inv(a)
    c[2] = inv(b)
    for s = 2:100
        c[s + 1] = c[s] * (a + s - 1) * inv(s * (b + s - 1))
    end
    
    return c
end
