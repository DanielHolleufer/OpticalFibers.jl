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
OpticalFibers._propagation_constant
OpticalFibers._propagation_constant_derivative
```

## Fibers
```@docs
radius
wavelength
frequency
material
refractive_index
propagation_constant
propagation_constant_derivative
normalized_frequency
effective_refractive_index
OpticalFibers.guided_mode_normalization_constant
```

## Electric Fields
```@docs
electric_guided_mode_cylindrical_base_components
electric_guided_mode_profile_cartesian_components
electric_guided_field_cartesian_components
electric_guided_field_cartesian_vector
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
transmission_two_level
transmission_three_level(Δes, fiber, Δr, Ωs::Number, gs, J, Γ, γ)
transmission_three_level(Δes, fiber, Δr, Ωs::AbstractArray, gs, J, Γ, γ)
```

## Atomic Traps
```@docs
gaussian_beam_intensity
tweezer_trap_intensity
tweezer_trap_potential
fiber_potential
full_potential
```
