abstract type TransmissionAlgorithm end

struct ContinuousPropagation <: TransmissionAlgorithm
    abstol::Float64
    reltol::Float64
    function ContinuousPropagation(abstol::Float64=1.0e-9, reltol::Float64=1.0e-9)
        return new(abstol, reltol)
    end
end

abstract type InputOutputAlgorithm <:TransmissionAlgorithm end

struct IndependentEmitters <: InputOutputAlgorithm end

struct CollectiveVacuumEmitters <: InputOutputAlgorithm end

struct CollectiveWaveguideEmitters <: InputOutputAlgorithm
    m_max::Int
    resolution::Int
    function CollectiveWaveguideEmitters(m_max::Int=64, resolution::Int=300)
        return new(m_max, resolution)
    end
end

abstract type ExperimentalSetup end

struct TwoLevelSetup{P<:GuidedMode,C<:AtomicCloud} <: ExperimentalSetup
    atom::TwoLevelAtom
    probe::P
    cloud::C
end

struct ThreeLevelSetup{P<:GuidedMode,C<:AtomicCloud,E<:ElectricField} <: ExperimentalSetup
    atom::ThreeLevelAtom
    probe::P
    cloud::C
    control::E
end

function experimental_setup(atom::TwoLevelAtom, probe::GuidedMode, cloud::AtomicCloud)
    return TwoLevelSetup(atom, probe, cloud)
end

function experimental_setup(
    atom::ThreeLevelAtom, probe::GuidedMode, cloud::AtomicCloud, control::ElectricField
)
    return ThreeLevelSetup(atom, probe, cloud, control)
end

function probe_detuning_range(setup::ThreeLevelSetup; kwargs...)
    return probe_detuning_range(
        setup.probe, setup.control, setup.atom, setup.cloud; kwargs...
    )
end

function transmission_coefficients(setup::ExperimentalSetup, alg::TransmissionAlgorithm)
    error("Selected algorithm is not implemented for selected setup.")
    return nothing
end

function transmission_coefficients(
    detunings::AbstractVector{<:Real},
    setup::ThreeLevelSetup{<:GuidedMode,GaussianCloud,<:ElectricField},
    alg::ContinuousPropagation
)
    probe = setup.probe
    control = setup.control
    atom = setup.atom
    cloud = setup.cloud

    abstol = alg.abstol
    reltol = alg.reltol

    ts = Vector{ComplexF64}(undef, length(detunings))
    for (i, Δ) in enumerate(detunings)
        ts[i] = transmission_coefficient_continuous_propagation(
            Δ, probe, control, atom, cloud, abstol, reltol
        )
    end

    return ts
end

function transmission_coefficients(
    detunings::AbstractVector{<:Real},
    setup::ThreeLevelSetup{<:GuidedMode,GaussianCloud,<:ElectricField},
    alg::InputOutputAlgorithm
)
    probe = setup.probe
    control = setup.control
    atom = setup.atom
    cloud = setup.cloud

    positions = atomic_cloud(cloud)
    gs = coupling_strengths(atom.lower_transition.dipole_moment, positions, probe)
    J, Γ = master_equation_coefficients(positions, atom.lower_transition, probe, alg)
    Ω = rabi_frequency(atom.upper_transition.dipole_moment, positions, control)
    H_effective = effective_hamiltonian(control, J, Γ)
    ts = transmission_three_level(detunings, probe, atom, gs, Ω, H_effective, Γ)

    return ts
end

function master_equation_coefficients(
    positions, transition, probe, ::IndependentEmitters
)
    fiber = probe.fiber
    J_guided, Γ_guided = guided_coefficients(positions, transition.dipole_moment, fiber)
    return J_guided, Γ_guided + 0.5 * transition.decay_rate * I
end

function master_equation_coefficients(
    positions, transition, probe, ::CollectiveVacuumEmitters
)
    fiber = probe.fiber
    ω₀ = frequency(fiber)
    J_guided, Γ_guided = guided_coefficients(positions, transition.dipole_moment, fiber)
    J_vacuum, Γ_vacuum = vacuum_coefficients(positions, transition.dipole_moment, ω₀)
    J_guided + J_vacuum, Γ_guided + Γ_vacuum
end

function master_equation_coefficients(
    positions, transition, probe, alg::CollectiveWaveguideEmitters
)
    fiber = probe.fiber
    m_max = alg.m_max
    resolution = alg.resolution
    J_guided, Γ_guided = guided_coefficients(positions, transition.dipole_moment, fiber)
    J_radiation, Γ_radiation = radiation_coefficients(
        positions, transition.dipole_moment, transition.decay_rate, fiber;
        m_max=m_max, resolution=resolution
    )
    return J_guided + J_radiation, Γ_guided + Γ_radiation
end

effective_hamiltonian(::ExternalField, J, Γ) = hessenberg(-(J + im * Γ / 2))
effective_hamiltonian(::GuidedField, J, Γ) = -(J + im * Γ / 2)
