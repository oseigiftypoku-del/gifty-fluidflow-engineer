"""
Gifty Fluidflow Engineer - Fluid & Material Properties Database
Provides standard fluid properties at representative conditions and pipe material roughness values.
"""

from typing import Dict, Any


# Standard representative fluid properties at ~20°C and 1 atm (unless noted)
PREDEFINED_FLUIDS: Dict[str, Dict[str, Any]] = {
    "Water (20°C)": {
        "density": 998.2,           # kg/m^3
        "dynamic_viscosity": 1.002e-3, # Pa*s (1.002 cP)
        "kinematic_viscosity": 1.004e-6, # m^2/s (1.004 cSt)
        "description": "Clean liquid water at 20°C, standard atmospheric pressure.",
        "category": "Liquid",
    },
    "Air (20°C, 1 atm)": {
        "density": 1.204,           # kg/m^3
        "dynamic_viscosity": 1.825e-5, # Pa*s (0.01825 cP)
        "kinematic_viscosity": 1.516e-5, # m^2/s
        "description": "Dry atmospheric air at 20°C and 101.325 kPa absolute pressure.",
        "category": "Gas",
    },
    "Engine Oil (SAE 30, 20°C)": {
        "density": 888.0,           # kg/m^3
        "dynamic_viscosity": 0.290,    # Pa*s (290 cP)
        "kinematic_viscosity": 3.266e-4, # m^2/s
        "description": "Representative automotive lubricant SAE 30 at 20°C.",
        "category": "Liquid",
    },
    "Crude Oil (Medium, 20°C)": {
        "density": 860.0,           # kg/m^3
        "dynamic_viscosity": 0.015,    # Pa*s (15 cP)
        "kinematic_viscosity": 1.744e-5, # m^2/s
        "description": "Representative 33° API medium crude oil at 20°C.",
        "category": "Liquid",
    },
    "Gasoline (20°C)": {
        "density": 740.0,           # kg/m^3
        "dynamic_viscosity": 6.0e-4,   # Pa*s (0.60 cP)
        "kinematic_viscosity": 8.108e-7, # m^2/s
        "description": "Standard automotive unleaded gasoline blend at 20°C.",
        "category": "Liquid",
    },
    "Kerosene (20°C)": {
        "density": 800.0,           # kg/m^3
        "dynamic_viscosity": 1.64e-3,  # Pa*s (1.64 cP)
        "kinematic_viscosity": 2.05e-6, # m^2/s
        "description": "Standard aviation/heating kerosene fuel at 20°C.",
        "category": "Liquid",
    },
    "Glycerin (20°C)": {
        "density": 1260.0,          # kg/m^3
        "dynamic_viscosity": 1.490,    # Pa*s (1490 cP)
        "kinematic_viscosity": 1.183e-3, # m^2/s
        "description": "Pure anhydrous glycerol at 20°C; highly viscous Newtonian fluid.",
        "category": "Liquid",
    },
    "Ethylene Glycol (50/50 Mix, 20°C)": {
        "density": 1065.0,          # kg/m^3
        "dynamic_viscosity": 3.5e-3,   # Pa*s (3.5 cP)
        "kinematic_viscosity": 3.286e-6, # m^2/s
        "description": "50% aqueous glycol automotive/industrial coolant blend.",
        "category": "Liquid",
    },
    "Seawater (20°C, 3.5% Salinity)": {
        "density": 1025.0,          # kg/m^3
        "dynamic_viscosity": 1.08e-3,  # Pa*s (1.08 cP)
        "kinematic_viscosity": 1.054e-6, # m^2/s
        "description": "Standard ocean seawater with 35 ppt salinity at 20°C.",
        "category": "Liquid",
    },
}


# Standard pipe material representative absolute roughness (epsilon) in meters
PIPE_ROUGHNESS_PRESETS: Dict[str, Dict[str, Any]] = {
    "Commercial Steel / Wrought Iron": {
        "roughness_m": 0.000045,      # 0.045 mm
        "roughness_mm": 0.045,
        "roughness_ft": 0.00015,
        "description": "Standard new welded/seamless commercial steel and wrought iron piping.",
    },
    "Stainless Steel (New)": {
        "roughness_m": 0.000015,      # 0.015 mm
        "roughness_mm": 0.015,
        "roughness_ft": 0.00005,
        "description": "Cold-drawn or pickled industrial stainless steel pipe.",
    },
    "PVC / Smooth Plastic / Glass": {
        "roughness_m": 0.0000015,     # 0.0015 mm
        "roughness_mm": 0.0015,
        "roughness_ft": 0.000005,
        "description": "Hydraulically smooth extruded thermoplastics, HDPE, PVC, and drawn glass.",
    },
    "Drawn Copper / Brass Tubing": {
        "roughness_m": 0.0000015,     # 0.0015 mm
        "roughness_mm": 0.0015,
        "roughness_ft": 0.000005,
        "description": "Seamless drawn copper, brass, or aluminum instrument tubing.",
    },
    "Galvanized Iron": {
        "roughness_m": 0.00015,       # 0.15 mm
        "roughness_mm": 0.15,
        "roughness_ft": 0.0005,
        "description": "Zinc hot-dip galvanized steel water pipe.",
    },
    "Cast Iron (Uncoated)": {
        "roughness_m": 0.00026,       # 0.26 mm
        "roughness_mm": 0.26,
        "roughness_ft": 0.00085,
        "description": "Standard sand-cast iron water and drainage pipe.",
    },
    "Concrete (Smooth Finish)": {
        "roughness_m": 0.00030,       # 0.30 mm
        "roughness_mm": 0.30,
        "roughness_ft": 0.0010,
        "description": "Precast structural concrete pipe with good steel formwork finish.",
    },
    "Riveted Steel / Heavily Corroded": {
        "roughness_m": 0.0030,        # 3.0 mm
        "roughness_mm": 3.0,
        "roughness_ft": 0.010,
        "description": "Aged, encrusted, tuberculated, or riveted penstocks.",
    },
}
