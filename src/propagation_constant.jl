"""
    fiber_equation(u, parameters)

Compute the value of the characteristic fiber equation with all terms moved to the same side
of the equal sign, where `u` is the propagation constant, and `parameters` contains the
fiber radius, refraction index, and frequency in said order.

Used as an input to the non-linear solver to find the propagation constant.

The characteristic fiber equation for single mode cylindrical fibers reads
[Fam Le Kien and A. Rauschenbeutel, Phys. Rev. A **95**, 023838 (2017)](https://link.aps.org/doi/10.1103/PhysRevA.90.063816)

```math
\\frac{J_{0}(p a)}{p a J_{1}(pa)} 
+ \\frac{n^{2} + 1}{2 n^{2}} \\frac{K_{1}'(q a)}{q a K_{1}(q a)} 
- \\frac{1}{p^2 a^2}
+ \\Biggl[\\biggl( \\frac{n^2 - 1}{2 n^2} \\frac{K_{1}'(q a)}{q a K_{1}(q a)} \\biggr)^2
+ \\frac{\\beta^2}{n^2 k^2} \\biggl( \\frac{1}{p^2 a^2} + \\frac{1}{q^2 a^2} \\biggr)^2
\\Biggr]^{1 / 2}
= 0,
```

where ``a`` is the fiber radius, ``k`` is the free space wave number of the light, ``n`` is
the refractive index of the fiber, ``p = \\sqrt{n^2 k^2 - \\beta^2}``, and
``q = \\sqrt{\\beta^2 - k^2}``. Futhermore, ``J_n`` and ``K_n`` are Bessel functions of the
first kind, and modified Bessel functions of the second kind, respectively, and the prime
denotes the derivative.
"""
function fiber_equation(u, parameters)
    a, n, ω = parameters
    p = sqrt(n^2 * ω^2 - u^2)
    q = sqrt(u^2 - ω^2)
    J0 = besselj0(p * a)
    J1 = besselj1(p * a)
    K1 = besselk1(q * a)
    K1p = -1 / 2 * (besselk0(q * a) + besselk(2, q * a))
    A = J0 / (p * a * J1)
    B = (n^2 + 1) / (2 * n^2) * K1p / (q * a * K1)
    C = -1 / ((p * a)^2)
    D1 = ((n^2 - 1) / (2 * n^2) * K1p / (q * a * K1))^2
    D2 = u^2 / (n^2 * ω^2) * (1 / ((q * a)^2) + 1 / ((p * a)^2))^2
    D = sqrt(D1 + D2)
    return A + B + C + D
end

"""
    propagation_constant(a::Real, n::Real, ω::Real)

Compute the propagation constant of a fiber with radius `a`, refactive index `n` and
frequency `ω` by solving the characteristic equation of the fiber as written as eq. (1A) in
[PRA 95, 023838].
"""
function propagation_constant(a::Real, n::Real, ω::Real)
    uspan = (ω + eps(ω), ω * n - eps(ω * n))
    parameters = (a, n, ω)
    prob_int = IntervalNonlinearProblem(fiber_equation, uspan, parameters)
    sol = solve(prob_int)
    return sol.u
end

"""
    propagation_constant_derivative(a::Real, n::Real, ω::Real; dω = 1e-9)

Compute the derivative of the propagation constant with respect to frequency evaluated at
`ω` of a fiber with radius `a`, and refactive index `n`.
"""
function propagation_constant_derivative(a::Real, n::Real, ω::Real; dω=1e-9)
    β_plus = propagation_constant(a, n, ω + dω / 2)
    β_minus = propagation_constant(a, n, ω - dω / 2)
    return (β_plus - β_minus) / dω
end
