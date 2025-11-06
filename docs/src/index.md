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
group_velocity
normalized_frequency
effective_refractive_index
OpticalFibers.guided_mode_normalization_constant
```

## Electric Fields
```@docs
electric_guided_mode_base
electric_guided_field_cartesian
```

## Master Equation Coefficients
```@docs
vacuum_coefficients
guided_coefficients
guided_directional_coefficients
radiation_coefficients
```

## Transmission
```@docs
transmission_two_level
transmission_three_level(Δes, fiber, Δr, Ωs::Number, gs, J, Γ, γ)
transmission_three_level(Δes, fiber, Δr, Ωs::AbstractArray, gs, J, Γ, γ)
```

## Atomic Traps
```@docs
AtomTrap
CrossedTweezerTrap
OpticalFibers.gaussian_beam_intensity
trap_intensity
trap_potential
```

## Atomic Clouds
```@docs
OpticalFibers.peak_density_gaussian_cloud
```