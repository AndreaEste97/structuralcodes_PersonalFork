"""Example of application of the TCM.
Starts from the official Structuralcodes example and continues."""

import numpy as np
import matplotlib.pyplot as plt
from shapely import Polygon

from structuralcodes import set_design_code
from structuralcodes.geometry import SurfaceGeometry, add_reinforcement
from structuralcodes.materials.concrete import create_concrete
from structuralcodes.materials.reinforcement import create_reinforcement
from structuralcodes.sections import BeamSection

set_design_code('MC2010')

# Create a concrete and a reinforcement
fck = 45
fyk = 450
ftk = 550
Es = 206000
epsuk = 0.007184466019417476

# These factory functions create concrete and reinforcement materials according
# to the globally set design code
concrete = create_concrete(fck=fck)
reinforcement = create_reinforcement(fyk=fyk, Es=Es, ftk=ftk, epsuk=epsuk)

# Create a rectangular geometry
width = 250
height = 500
polygon = Polygon(
    [
        (-width / 2, -height / 2),
        (width / 2, -height / 2),
        (width / 2, height / 2),
        (-width / 2, height / 2),
    ]
)  # We leverage shapely to create geometries
geometry = SurfaceGeometry(
    poly=polygon, material=concrete
)  # A SurfaceGeometry is a shapely Polygon with an assigned material

# Add reinforcement
diameter_reinf = 25
cover = 50

geometry = add_reinforcement(
    geometry,
    (
        -width / 2 + cover + diameter_reinf / 2,
        -height / 2 + cover + diameter_reinf / 2,
    ),
    diameter_reinf,
    reinforcement,
)  # The add_reinforcement function returns a CompoundGeometry
geometry = add_reinforcement(
    geometry,
    (
        width / 2 - cover - diameter_reinf / 2,
        -height / 2 + cover + diameter_reinf / 2,
    ),
    diameter_reinf,
    reinforcement,
)
geometry = add_reinforcement(
    geometry,
    (
        -width / 2 + cover + diameter_reinf / 2,
        height / 2 - cover - diameter_reinf / 2,
    ),
    diameter_reinf,
    reinforcement,
)
geometry = add_reinforcement(
    geometry,
    (
        width / 2 - cover - diameter_reinf / 2,
        height / 2 - cover - diameter_reinf / 2,
    ),
    diameter_reinf,
    reinforcement,
)

# Create section
section = BeamSection(geometry)

Fmax = section.section_calculator.calculate_limit_axial_load()
Fmax_Tensile = Fmax[1]
#print(Fmax_Tensile)




#-----------------------------------------------Applicazione TCM----------------------------------------------------

#Additional iput
As=section.gross_properties.area_reinforcement    #Area reinforcement
Ac=section.gross_properties.area-As               #Area concrete
Stress_failure_concrete=5                         #ultimate stress of the concrete [MPa]
Young_modulus_concrete=34000                      #Young modulus of the concrete [MPa]
Tau0=1                                            #shear stress between reinforcement and steel before yielding of the steel [MPa]
Tau1=0.5                                          #shear stress between reinforcement and steel after yielding of the steel [MPa]

#application
F_array=np.linspace(0,Fmax_Tensile,1000)
Eps_array=section.section_calculator.get_axial_strain_according_to_TCM(force_array=F_array,stress_failure_concrete=Stress_failure_concrete,
                                                                       young_modulus_concrete=Young_modulus_concrete,tau_0=Tau0,tau_1=Tau1)

#plot
fig, ax = plt.subplots()
ax.plot(Eps_array, F_array, '-k')
ax.grid()
ax.set_xlabel('epsilon$ [-]')
ax.set_ylabel('F$ [N]')

#optional: add naked steel curve on graph, only for comparison
def GetEpsOnSteel(ForceOnSteel):
    Sigma=ForceOnSteel/As
    if Sigma<=fyk:
        Eps=Sigma/Es
    else:
        Eps=fyk/Es+(Sigma-fyk)/((ftk-fyk)/(epsuk-fyk/Es))
    return Eps
Strain_NakedSteel=np.array([GetEpsOnSteel(force) for force in F_array])
ax.plot(Strain_NakedSteel, F_array, '-k')

plt.show()