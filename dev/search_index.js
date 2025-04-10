var documenterSearchIndex = {"docs":
[{"location":"#OpticalFibers.jl","page":"Home","title":"OpticalFibers.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for OpticalFibers.jl","category":"page"},{"location":"#Materials","page":"Home","title":"Materials","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Material\nOpticalFibers.sellmeier_equation","category":"page"},{"location":"#OpticalFibers.Material","page":"Home","title":"OpticalFibers.Material","text":"Material(B1::Real, B2::Real, B3::Real, C1::Real, C2::Real, C3::Real)\n\nCreate a Material with Sellmeier coefficients B₁, B₂, B₃, C₁, C₂, and C₃, where C₁, C₂, and C₃ must be given in square micrometers.\n\n\n\n\n\n","category":"type"},{"location":"#OpticalFibers.sellmeier_equation","page":"Home","title":"OpticalFibers.sellmeier_equation","text":"sellmeier_equation(material::Material, wavelength_μm::Real)\n\nCompute the refractive index, n, of the material for light with a free space wavelength of wavelength_μm, which must be given in micrometers.\n\nThe Sellmeier equation reads\n\nn^2(λ) = 1 + sum_i fracB_i λ^2λ^2 - C_i\n\nwhere B_i and C_i are constants that depend on the material, and λ is the wavelength of light in vacuum. \n\n\n\n\n\n","category":"function"},{"location":"#Propagation-Constant","page":"Home","title":"Propagation Constant","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"OpticalFibers.fiber_equation\nOpticalFibers._propagation_constant\nOpticalFibers._propagation_constant_derivative","category":"page"},{"location":"#OpticalFibers.fiber_equation","page":"Home","title":"OpticalFibers.fiber_equation","text":"fiber_equation(u, parameters)\n\nCompute the value of the characteristic fiber equation with all terms moved to the same side of the equal sign, where u is the propagation constant, and parameters contains the fiber radius, refraction index, and frequency in said order.\n\nUsed as an input to the non-linear solver to find the propagation constant.\n\nThe characteristic fiber equation for single mode cylindrical fibers reads Fam Le Kien and A. Rauschenbeutel, Phys. Rev. A 95, 023838 (2017)\n\nfracJ_0(p a)p a J_1(pa) \n+ fracn^2 + 12 n^2 fracK_1(q a)q a K_1(q a) \n- frac1p^2 a^2\n+ Bigglbiggl( fracn^2 - 12 n^2 fracK_1(q a)q a K_1(q a) biggr)^2\n+ fracbeta^2n^2 k^2 biggl( frac1p^2 a^2 + frac1q^2 a^2 biggr)^2\nBiggr^1  2\n= 0\n\nwhere a is the fiber radius, k is the free space wave number of the light, n is the refractive index of the fiber, p = sqrtn^2 k^2 - beta^2, and q = sqrtbeta^2 - k^2. Futhermore, J_n and K_n are Bessel functions of the first kind, and modified Bessel functions of the second kind, respectively, and the prime denotes the derivative.\n\n\n\n\n\n","category":"function"},{"location":"#OpticalFibers._propagation_constant","page":"Home","title":"OpticalFibers._propagation_constant","text":"_propagation_constant(a::Real, n::Real, ω::Real)\n\nCompute the propagation constant of a fiber with radius a, refactive index n and frequency ω by solving the characteristic equation of the fiber as written as eq. (1A) in [PRA 95, 023838].\n\n\n\n\n\n","category":"function"},{"location":"#OpticalFibers._propagation_constant_derivative","page":"Home","title":"OpticalFibers._propagation_constant_derivative","text":"_propagation_constant_derivative(a::Real, n::Real, ω::Real; dω = 1e-9)\n\nCompute the derivative of the propagation constant with respect to frequency evaluated at ω of a fiber with radius a, and refactive index n.\n\n\n\n\n\n","category":"function"},{"location":"#Fibers","page":"Home","title":"Fibers","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"radius\nwavelength\nfrequency\nmaterial\nrefractive_index\npropagation_constant\npropagation_constant_derivative\nnormalized_frequency\nOpticalFibers.guided_mode_normalization_constant","category":"page"},{"location":"#OpticalFibers.radius","page":"Home","title":"OpticalFibers.radius","text":"radius(fiber::Fiber)\n\nReturn the radius of the fiber in micrometers.\n\nExamples\n\njulia> fiber = Fiber(0.1, 0.4, Material(0.6961663, 0.4079426, 0.8974794, 0.0684043^2, 0.1162414^2, 9.896161^2));\n\njulia> radius(fiber)\n0.1\n\n\n\n\n\n","category":"function"},{"location":"#OpticalFibers.wavelength","page":"Home","title":"OpticalFibers.wavelength","text":"wavelength(fiber::Fiber)\n\nReturn the wavelength of the fiber mode in micrometers.\n\nExamples\n\njulia> fiber = Fiber(0.1, 0.4, Material(0.6961663, 0.4079426, 0.8974794, 0.0684043^2, 0.1162414^2, 9.896161^2));\n\njulia> wavelength(fiber)\n0.4\n\n\n\n\n\n","category":"function"},{"location":"#OpticalFibers.frequency","page":"Home","title":"OpticalFibers.frequency","text":"frequency(fiber::Fiber)\n\nReturn the frequency of the fiber mode.\n\n\n\n\n\n","category":"function"},{"location":"#OpticalFibers.material","page":"Home","title":"OpticalFibers.material","text":"material(fiber::Fiber)\n\nReturn the material of the fiber.\n\n\n\n\n\n","category":"function"},{"location":"#OpticalFibers.refractive_index","page":"Home","title":"OpticalFibers.refractive_index","text":"refractive_index(fiber::Fiber)\n\nReturn the refractive index of the fiber for light with the same wavelength as the fiber mode.\n\n\n\n\n\n","category":"function"},{"location":"#OpticalFibers.propagation_constant","page":"Home","title":"OpticalFibers.propagation_constant","text":"propagation_constant(fiber::Fiber)\n\nReturn the propagation constant of the fiber for light with the same wavelength as the fiber mode.\n\n\n\n\n\n","category":"function"},{"location":"#OpticalFibers.propagation_constant_derivative","page":"Home","title":"OpticalFibers.propagation_constant_derivative","text":"propagation_constant_derivative(fiber::Fiber)\n\nReturn the derivative of the propagation constant of the fiber evaluated at the the wavelength of the fiber mode.\n\n\n\n\n\n","category":"function"},{"location":"#OpticalFibers.normalized_frequency","page":"Home","title":"OpticalFibers.normalized_frequency","text":"normalized_frequency(fiber::Fiber)\n\nReturn the normalized frequency of the fiber mode.\n\n\n\n\n\n","category":"function"},{"location":"#OpticalFibers.guided_mode_normalization_constant","page":"Home","title":"OpticalFibers.guided_mode_normalization_constant","text":"guided_mode_normalization_constant(a::Real, n::Real, β::Real, h::Real, q::Real, K1J1::Real, s::Real)\n\nCompute the normalization constant of an electric guided fiber mode.\n\nThe fiber modes are normalized according to the condition\n\nint_0^infty  mathrmd rho int_0^2 pi  mathrmd phi  \nn^2(rho) lvert mathrmmathbfe(rho phi) rvert^2 = 1\n\nwhere n(rho) is the step index refractive index given as\n\nn(rho) = begincases\n    n  rho  a \n    1  rho  a\nendcases\n\nand mathrmmathbfe(rho phi) = e_rho hatmathrmmathbfrho + e_phi hatmathrmmathbfphi + e_z hatmathrmmathbfz, where the components are given by electric_guided_mode_cylindrical_base_components.\n\n\n\n\n\n","category":"function"},{"location":"#Electric-Fields","page":"Home","title":"Electric Fields","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"electric_guided_mode_cylindrical_base_components\nelectric_guided_mode_profile_cartesian_components\nelectric_guided_field_cartesian_components\nelectric_guided_field_cartesian_vector","category":"page"},{"location":"#OpticalFibers.electric_guided_mode_cylindrical_base_components","page":"Home","title":"OpticalFibers.electric_guided_mode_cylindrical_base_components","text":"electric_guided_mode_cylindrical_base_components(ρ::Real, a::Real, β::Real, p::Real, q::Real, K1J1::Real, s::Real)\n\nCompute the underlying cylindrical components of the guided mode electric field used in the expressions for both the quasilinear and quasicircular guided modes.\n\nThese components for rho  a are given by\n\nbeginaligned\n    e_rho = A mathrmi fracqp fracK_1(q a)J_1(p a) (1 - s) J_0(p rho) - (1 + s) J_2(p rho) \n    e_phi = -A fracqp fracK_1(q a)J_1(p a) (1 - s) J_0(p rho) + (1 + s) J_2(p rho) \n    e_z = A frac2 qbeta fracK_1(q a)J_1(p a) J_1(p rho)\nendaligned\n\nand the components for rho  a are given by\n\nbeginaligned\n    e_rho = A mathrmi (1 - s) K_0(q rho) + (1 + s) K_2(q rho) \n    e_phi = -A (1 - s) K_0(q rho) - (1 + s) K_2(q rho) \n    e_z = A frac2 qbeta K_1(q rho)\nendaligned\n\nwhere A is the normalization constant, a is the fiber radius, beta is the propagation constant, p = sqrtn^2 k^2 - beta^2, and q = sqrtbeta^2 - k^2, with k being the free space wavenumber of the light. Futhermore, J_n and K_n are Bessel functions of the first kind, and modified Bessel functions of the second kind, respectively, and the prime denotes the derivative. Lastly, s is defined as\n\ns = fracfrac1p^2 a^2 + frac1q^2 a^2fracJ_1(p a)p a J_1(p a) + fracK_1(q a)q a K_1(q a)\n\n\n\n\n\n","category":"function"},{"location":"#OpticalFibers.electric_guided_mode_profile_cartesian_components","page":"Home","title":"OpticalFibers.electric_guided_mode_profile_cartesian_components","text":"electric_guided_mode_profile_cartesian_components(ρ::Real, ϕ::Real, l::Integer, f::Integer, fiber::Fiber, ::CircularPolarization)\n\nCompute the cartesian components of the mode profile of the guided electric field in the plane transverse to the fiber.\n\nThe components are given by\n\nbeginaligned\n    e_x = (e_rho cos(phi) - l e_phi sin(phi)) mathrme^mathrmi l phi \n    e_y = (e_rho sin(phi) + l e_phi cos(phi)) mathrme^mathrmi l phi \n    e_z = f e_z mathrme^mathrmi l phi\nendaligned\n\nwhere A is the normalization constant, l is the polarization, and f is the direction of propagation. The base components, e_rho, e_phi, and e_rho, are given by electric_guided_mode_cylindrical_base_components.\n\n\n\n\n\n","category":"function"},{"location":"#OpticalFibers.electric_guided_field_cartesian_components","page":"Home","title":"OpticalFibers.electric_guided_field_cartesian_components","text":"electric_guided_field_cartesian_components(ρ::Real, ϕ::Real, l::Integer, f::Integer, fiber::Fiber, polarization_basis::CircularPolarization, power::Real)\n\nCompute the cartesian components of a guided electric field at position (ρ ϕ z) and time t with the given power.\n\n\n\n\n\n","category":"function"},{"location":"#OpticalFibers.electric_guided_field_cartesian_vector","page":"Home","title":"OpticalFibers.electric_guided_field_cartesian_vector","text":"electric_guided_field_cartesian_vector(ρ::Real, ϕ::Real, l::Integer, f::Integer, fiber::Fiber, polarization_basis::CircularPolarization, power::Real)\n\nCompute the guided electric field vector at position (ρ ϕ z) and time t with the given power.\n\n\n\n\n\n","category":"function"},{"location":"#Master-Equation-Coefficients","page":"Home","title":"Master Equation Coefficients","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"vacuum_coefficients\nguided_mode_coefficients\nguided_mode_directional_coefficients\nradiation_mode_decay_coefficients\nradiation_mode_coefficients\nradiation_mode_directional_coefficients\nOpticalFibers.greens_function_free_space\nOpticalFibers.guided_coupling_strength\nOpticalFibers.radiative_coupling_strength","category":"page"},{"location":"#OpticalFibers.vacuum_coefficients","page":"Home","title":"OpticalFibers.vacuum_coefficients","text":"vacuum_coefficients(r, d, ω₀)\n\nCompute the dipole-dipole and decay coefficients for the master equation describing a cloud of atoms with positions given by the columns in r (in cartesian coordinates), dipole moment d, and transition frequency ω₀ coupled to the vacuum field.\n\n\n\n\n\n","category":"function"},{"location":"#OpticalFibers.guided_mode_coefficients","page":"Home","title":"OpticalFibers.guided_mode_coefficients","text":"guided_mode_coefficients(r, d, fiber)\n\nCompute the guided dipole-dipole and decay coefficients for the master equation describing a cloud of atoms with positions given by the columns in r (in cartesian coordinates), and dipole moment d coupled to an optical fiber.\n\n\n\n\n\n","category":"function"},{"location":"#OpticalFibers.guided_mode_directional_coefficients","page":"Home","title":"OpticalFibers.guided_mode_directional_coefficients","text":"guided_mode_directional_coefficients(r, d, fiber)\n\nCompute the guided dipole-dipole and decay coefficients due to the modes with direction f for the master equation describing a cloud of atoms with positions given by the columns in r (in cartesian coordinates), and dipole moment d coupled to an optical fiber.\n\n\n\n\n\n","category":"function"},{"location":"#OpticalFibers.radiation_mode_decay_coefficients","page":"Home","title":"OpticalFibers.radiation_mode_decay_coefficients","text":"radiation_mode_decay_coefficients(r, d, fiber, polarization_basis::CircularPolarization; abstol = 1e-3)\n\nCompute the decay coefficients for the master equation describing a cloud of atoms with positions given by the columns in r (in cartesian coordinates), and dipole moment d coupled to the radiation modes from an optical fiber.\n\n\n\n\n\n","category":"function"},{"location":"#OpticalFibers.radiation_mode_coefficients","page":"Home","title":"OpticalFibers.radiation_mode_coefficients","text":"radiation_mode_coefficients(r, d, fiber, polarization_basis::CircularPolarization; abstol = 1e-3)\n\nCompute the dipole-dipole and decay coefficients for the master equation describing a cloud of atoms with positions given by the columns in r (in cartesian coordinates), and dipole moment d coupled to the radiation modes from an optical fiber.\n\n\n\n\n\n","category":"function"},{"location":"#OpticalFibers.radiation_mode_directional_coefficients","page":"Home","title":"OpticalFibers.radiation_mode_directional_coefficients","text":"radiation_mode_directional_coefficients(r, d, fiber, polarization_basis::CircularPolarization, f; abstol = 1e-3)\n\nCompute the dipole-dipole and decay coefficients for the master equation describing a cloud of atoms with positions given by the columns in r (in cartesian coordinates), and dipole moment d coupled to the radiation modes with directions f from an optical fiber.\n\n\n\n\n\n","category":"function"},{"location":"#OpticalFibers.greens_function_free_space","page":"Home","title":"OpticalFibers.greens_function_free_space","text":"greens_function_free_space(k, r)\n\nCompute the Green's tensor of the electric field in free space with wave number k and position vector k.\n\n\n\n\n\n","category":"function"},{"location":"#OpticalFibers.guided_coupling_strength","page":"Home","title":"OpticalFibers.guided_coupling_strength","text":"guided_coupling_strength(ρ, ϕ, z, d, l, f, fiber)\n\nCompute the coupling strength between an atom and a guided fiber mode.\n\nImplementation of Eq. (7), top equation from Fam Le Kien and A. Rauschenbeutel.  \"Nanofiber-mediated chiral radiative coupling between two atoms\". Phys. Rev. A 95, 023838 (2017).\n\n\n\n\n\n","category":"function"},{"location":"#OpticalFibers.radiative_coupling_strength","page":"Home","title":"OpticalFibers.radiative_coupling_strength","text":"radiative_coupling_strength(ρ, ϕ, z, d, l, f, fiber, polarization_basis::CircularPolarization)\n\nCompute the coupling strength between an atom and a radiation fiber mode.\n\nImplementation of Eq. (7), bottom equation from Fam Le Kien and A. Rauschenbeutel.  \"Nanofiber-mediated chiral radiative coupling between two atoms\". Phys. Rev. A 95, 023838 (2017).\n\n\n\n\n\n","category":"function"},{"location":"#Transmission","page":"Home","title":"Transmission","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"transmission_three_level(Δes, fiber, Δr, Ωs::Number, gs, J, Γ, γ)\ntransmission_three_level(Δes, fiber, Δr, Ωs::AbstractArray, gs, J, Γ, γ)","category":"page"},{"location":"#OpticalFibers.transmission_three_level-Tuple{Any, Any, Any, Number, Vararg{Any, 4}}","page":"Home","title":"OpticalFibers.transmission_three_level","text":"transmission_three_level(Δes, fiber, Δr, Ω::Number, gs, J, Γ, γ)\n\nCompute the transmission of a cloud of three level atoms surrounding an optical fiber for  each value of the lower transition detuning given by Δes, where each atom experience the same Rabi frequency.\n\nThe parameters of the fiber are given by fiber, while the atoms have upper transition detuning Δr, control Rabi frequenciy Ω, pump coupling constants gs, dipole-dipole interaction matrix J, cross decay rate matrix Γ, and Rydberg to intermediate state decay rate γ.\n\n\n\n\n\n","category":"method"},{"location":"#OpticalFibers.transmission_three_level-Tuple{Any, Any, Any, AbstractArray, Vararg{Any, 4}}","page":"Home","title":"OpticalFibers.transmission_three_level","text":"transmission_three_level(Δes, fiber, Δr, Ωs::AbstractArray, gs, J, Γ, γ)\n\nCompute the transmission of a cloud of three level atoms surrounding an optical fiber for  each value of the lower transition detuning given by Δes, where the Rabi frequency can be different from atom to atom.\n\nThe parameters of the fiber are given by fiber, while the atoms have upper transition detuning Δr, control Rabi frequencies Ωs, pump coupling constants gs, dipole-dipole interaction matrix J, cross decay rate matrix Γ, and Rydberg to intermediate state decay rate γ.\n\n\n\n\n\n","category":"method"}]
}
