# This package uses a system of natural units where c = ħ = k_B = ϵ₀ = μ₀ = 1,
# and where lengths are measured in micrometers, μm. Here we provide the conversion
# factors between SI units and natural units.
#
# For distance the conversion factor is just the conversion from meters to micrometers
#   meter = 10^6
# 
# To get the remaining SI base units of interest, second, kilogram, ampere, and kelvin,
# we solve for them in the following equations
#   c = 2.99792458e8 * meter / second = 1
#   ħ = 6.62607015e-34 / 2π * kilogram * meter^2 / second = 1
#   k_B = 1.380649e-23 * kilogram * measured^2 / (kelvin * second^2) = 1
#   ε₀ = 8.8541878188e-12 * ampere^2 * second^4 / (kilogram * meter^3) = 1
#
# The solutions are
#   second = c * meter
#   kilogram = second / (ħ * meter^2)
#   ampere = sqrt(meter^3 * kilogram / (ε₀ * second^4))
#   kelvin = k_B * kilogram * meter^2 / (second^2)
#
# The remaining conversion factors are calculated using the SI base unit conversion factors.
#   unit of area = meter^2
#   unit of volume = meter^3
#   hertz = 1 / second
#   unit of velocity = meter / second
#   unit of acceleration = meter / (second^2)
#   unit of momentum = kilogram * meter / second
#   newton = kilogram * meter / (second^2)
#   joule = newton * meter
#   watt = joule / second
#   unit of intensity = watt / (meter^2)
#   coulomb = ampere * second
#   unit of electric field = newton / coulomb
#   unit of electric diople moment = meter * coulomb
#   farad = ampere^2 * second^4 / (kilogram * meter^2)
#   permitivity = farad / meter
#   permeability = newton / (ampere^2)
# 
# To get the conversion rates from natural units to SI units, we just take the inverse of
# the conversion rates from SI to natural.

struct UnitConversions
    length::Float64
    time::Float64
    mass::Float64
    current::Float64
    temperature::Float64
    area::Float64
    volume::Float64
    frequency::Float64
    velocity::Float64
    acceleration::Float64
    momentum::Float64
    force::Float64
    energy::Float64
    power::Float64
    intensity::Float64
    charge::Float64
    electric_field::Float64
    electric_dipole_moment::Float64
    capacitance::Float64
    permitivity::Float64
    permeability::Float64
end

function UnitConversions(meter, c, ħ, k_B, ε₀)
    second = c * meter
    kilogram = second / (ħ * meter^2)
    ampere = sqrt(meter^3 * kilogram / (ε₀ * second^4))
    kelvin = k_B * kilogram * meter^2 / (second^2)

    area = meter^2
    volume = meter^3
    hertz = 1 / second
    velocity = meter / second
    acceleration = meter / (second^2)
    momentum = kilogram * meter / second
    newton = kilogram * meter / (second^2)
    joule = newton * meter
    watt = joule / second
    intensity = watt / (meter^2)
    coulomb = ampere * second
    electric_field = newton / coulomb
    electric_diople_moment = meter * coulomb
    farad = ampere^2 * second^4 / (kilogram * meter^2)
    permitivity = farad / meter
    permeability = newton / (ampere^2)
    
    return UnitConversions(
        meter,
        second,
        kilogram,
        ampere,
        kelvin,
        area,
        volume,
        hertz,
        velocity,
        acceleration,
        momentum,
        newton,
        joule,
        watt,
        intensity,
        coulomb,
        electric_field,
        electric_diople_moment,
        farad,
        permitivity,
        permeability,
    )
end

const SI_to_natural = UnitConversions(
    1.0e6,
    2.99792458e14,
    2.842788447250069e36,
    6304.58493631402,
    0.00043670324463929307,
    1.0e12,
    1.0e18,
    3.3356409519815205e-15,
    3.3356409519815204e-9,
    1.1126500560536183e-23,
    9.482521562467288e27,
    3.163028725181368e13,
    3.163028725181368e19,
    105507.28148008873,
    1.0550728148008873e-7,
    1.8900670147273533e18,
    1.6735008338514613e-5,
    1.8900670147273533e24,
    1.1294090666076803e17,
    1.1294090666076804e11,
    795774.7155654947
)

const natural_to_SI = UnitConversions(
    1.0e-6,
    3.3356409519815205e-15,
    3.5176729417461077e-37,
    0.00015861472406217606,
    2289.884520610735,
    1.0e-12,
    1.0e-18,
    2.99792458e14,
    2.99792458e8,
    8.987551787368177e22,
    1.0545718176461564e-28,
    3.16152677349669e-14,
    3.1615267734966905e-20,
    9.47801882459382e-6,
    9.478018824593822e6,
    5.290817691690431e-19,
    59754.97470763491,
    5.290817691690431e-25,
    8.8541878188e-18,
    8.854187818799999e-12,
    1.2566370612685004e-6
)
