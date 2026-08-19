"""
Gifty Fluidflow Engineer - Unit Conversion Engine
Converts all user inputs into internal SI base units before calculations,
and formats results back into the requested display unit system.
"""

from typing import Dict, Any


class UnitConverter:
    """Rigorous unit converter ensuring internal SI calculation consistency."""

    # Length / Diameter
    @staticmethod
    def length_to_si(val: float, unit: str) -> float:
        u = unit.lower().strip()
        if u in ["m", "meter", "meters"]:
            return val
        elif u in ["mm", "millimeter", "millimeters"]:
            return val * 1e-3
        elif u in ["cm", "centimeter"]:
            return val * 1e-2
        elif u in ["ft", "feet", "foot"]:
            return val * 0.3048
        elif u in ["in", "inch", "inches"]:
            return val * 0.0254
        return val

    @staticmethod
    def length_from_si(val: float, unit: str) -> float:
        u = unit.lower().strip()
        if u in ["m", "meter", "meters"]:
            return val
        elif u in ["mm", "millimeter", "millimeters"]:
            return val * 1e3
        elif u in ["cm", "centimeter"]:
            return val * 1e2
        elif u in ["ft", "feet", "foot"]:
            return val / 0.3048
        elif u in ["in", "inch", "inches"]:
            return val / 0.0254
        return val

    # Density
    @staticmethod
    def density_to_si(val: float, unit: str) -> float:
        u = unit.lower().strip()
        if u in ["kg/m3", "kg/m^3"]:
            return val
        elif u in ["g/cm3", "g/cm^3", "specific_gravity"]:
            return val * 1000.0
        elif u in ["lbm/ft3", "lbm/ft^3", "lb/ft3", "lb/ft^3"]:
            return val * 16.018463
        return val

    @staticmethod
    def density_from_si(val: float, unit: str) -> float:
        u = unit.lower().strip()
        if u in ["kg/m3", "kg/m^3"]:
            return val
        elif u in ["g/cm3", "g/cm^3"]:
            return val / 1000.0
        elif u in ["lbm/ft3", "lbm/ft^3", "lb/ft3", "lb/ft^3"]:
            return val / 16.018463
        return val

    # Dynamic Viscosity
    @staticmethod
    def dynamic_viscosity_to_si(val: float, unit: str) -> float:
        u = unit.lower().strip()
        if u in ["pa*s", "pa.s", "pa s", "n*s/m2", "kg/(m*s)"]:
            return val
        elif u in ["cp", "centipoise", "mpa*s", "mpa.s"]:
            return val * 1e-3
        elif u in ["p", "poise"]:
            return val * 0.1
        elif u in ["lbf*s/ft2", "lbf.s/ft2", "slug/(ft*s)"]:
            return val * 47.880259
        elif u in ["lbm/(ft*s)"]:
            return val * 1.4881639
        return val

    @staticmethod
    def dynamic_viscosity_from_si(val: float, unit: str) -> float:
        u = unit.lower().strip()
        if u in ["pa*s", "pa.s", "pa s"]:
            return val
        elif u in ["cp", "centipoise", "mpa*s"]:
            return val * 1e3
        elif u in ["p", "poise"]:
            return val * 10.0
        elif u in ["lbf*s/ft2", "lbf.s/ft2"]:
            return val / 47.880259
        elif u in ["lbm/(ft*s)"]:
            return val / 1.4881639
        return val

    # Volumetric Flow Rate
    @staticmethod
    def flow_rate_to_si(val: float, unit: str) -> float:
        u = unit.lower().strip()
        if u in ["m3/s", "m^3/s", "cumecs"]:
            return val
        elif u in ["l/s", "liter/s", "liters/s"]:
            return val * 1e-3
        elif u in ["l/min", "lpm"]:
            return val * (1e-3 / 60.0)
        elif u in ["m3/h", "m^3/h", "m3/hr"]:
            return val / 3600.0
        elif u in ["ft3/s", "ft^3/s", "cfs"]:
            return val * 0.028316847
        elif u in ["ft3/min", "ft^3/min", "cfm"]:
            return val * (0.028316847 / 60.0)
        elif u in ["gpm", "gal/min", "us gpm"]:
            return val * 0.0000630901964
        return val

    @staticmethod
    def flow_rate_from_si(val: float, unit: str) -> float:
        u = unit.lower().strip()
        if u in ["m3/s", "m^3/s"]:
            return val
        elif u in ["l/s", "liter/s"]:
            return val * 1e3
        elif u in ["l/min", "lpm"]:
            return val * 60000.0
        elif u in ["m3/h", "m^3/h"]:
            return val * 3600.0
        elif u in ["ft3/s", "ft^3/s", "cfs"]:
            return val / 0.028316847
        elif u in ["ft3/min", "ft^3/min", "cfm"]:
            return val / (0.028316847 / 60.0)
        elif u in ["gpm", "gal/min", "us gpm"]:
            return val / 0.0000630901964
        return val

    # Velocity
    @staticmethod
    def velocity_to_si(val: float, unit: str) -> float:
        u = unit.lower().strip()
        if u in ["m/s", "mps"]:
            return val
        elif u in ["km/h", "kph"]:
            return val / 3.6
        elif u in ["ft/s", "fps"]:
            return val * 0.3048
        elif u in ["mph"]:
            return val * 0.44704
        return val

    @staticmethod
    def velocity_from_si(val: float, unit: str) -> float:
        u = unit.lower().strip()
        if u in ["m/s", "mps"]:
            return val
        elif u in ["km/h", "kph"]:
            return val * 3.6
        elif u in ["ft/s", "fps"]:
            return val / 0.3048
        elif u in ["mph"]:
            return val / 0.44704
        return val

    # Pressure
    @staticmethod
    def pressure_to_si(val: float, unit: str) -> float:
        u = unit.lower().strip()
        if u in ["pa", "pascal", "n/m2"]:
            return val
        elif u in ["kpa", "kilopascal"]:
            return val * 1e3
        elif u in ["mpa", "megapascal"]:
            return val * 1e6
        elif u in ["bar"]:
            return val * 1e5
        elif u in ["mbar"]:
            return val * 100.0
        elif u in ["psi", "lbf/in2", "lbf/in^2"]:
            return val * 6894.75729
        elif u in ["psf", "lbf/ft2"]:
            return val * 47.880259
        elif u in ["atm", "atmosphere"]:
            return val * 101325.0
        return val

    @staticmethod
    def pressure_from_si(val: float, unit: str) -> float:
        u = unit.lower().strip()
        if u in ["pa", "pascal"]:
            return val
        elif u in ["kpa", "kilopascal"]:
            return val / 1e3
        elif u in ["mpa", "megapascal"]:
            return val / 1e6
        elif u in ["bar"]:
            return val / 1e5
        elif u in ["mbar"]:
            return val / 100.0
        elif u in ["psi", "lbf/in2"]:
            return val / 6894.75729
        elif u in ["psf"]:
            return val / 47.880259
        elif u in ["atm"]:
            return val / 101325.0
        return val
