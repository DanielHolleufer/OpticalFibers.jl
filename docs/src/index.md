# OpticalFibers.jl

Documentation for OpticalFibers.jl

## Materials
```@docs
Material
OpticalFibers.sellmeier_equation
```

## Propagation Constant
```@docs
OpticalFibers.fiber_equation
propagation_constant(a::Real, n::Real, ω::Real)
propagation_constant_derivative(a::Real, n::Real, ω::Real)
```

## Fiber Modes
```@docs
radius
wavelength
frequency
material
refractive_index
propagation_constant
propagation_constant_derivative(fiber::Fiber)
normalized_frequency(fiber::Fiber)
OpticalFibers.electric_guided_mode_cylindrical_components_unnormalized
OpticalFibers.electric_guided_mode_normalization_constant
```

## Master Equation Coefficients
```@docs
vacuum_coefficients
guided_mode_coefficients
guided_mode_directional_coefficients
radiation_mode_decay_coefficients
radiation_mode_coefficients
radiation_mode_directional_coefficients
OpticalFibers.greens_function_free_space
OpticalFibers.guided_coupling_strength
OpticalFibers.radiative_coupling_strength
```

## Transmission
```@docs
transmission_three_level(Δes, fiber, Δr, Ωs::Number, gs, J, Γ, γ)
transmission_three_level(Δes, fiber, Δr, Ωs::AbstractArray, gs, J, Γ, γ)
```
