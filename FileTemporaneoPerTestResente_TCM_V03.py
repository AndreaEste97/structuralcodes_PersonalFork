#V03: implemented average Lambda=0.67 instead of 1.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

#reinforcement consitutive law (bilinear)
Stress_yield_reinf=450         #yield stress of the reinforcement [MPa]
Stress_failure_reinf=550       #ultimate stress of the reinforcement [MPa]
Young_modulus_reinf=206000     #Young modulus of the reinforcement [MPa]
Strain_yield_reinf=Stress_yield_reinf/Young_modulus_reinf #yield strain of the reinforcement [-]
Post_yield_modulus_reinf=20000 #Tangent modulus of the reinforcement after the yield [MPa]
Strain_failure_reinf=Strain_yield_reinf+(Stress_failure_reinf-Stress_yield_reinf)/Post_yield_modulus_reinf #ultimate strain of the reinforcement [-]
print(Strain_failure_reinf)

#concrete consitutive law in tension (linear)
Stress_failure_concrete=5         #ultimate stress of the concrete [MPa]
Young_modulus_concrete=34000     #Young modulus of the concrete [MPa]
Strain_failure_concrete=Stress_failure_concrete/Young_modulus_concrete #ultimate strain of the reinforcement [-]

if False:
    strains_reinf = np.linspace(0,Strain_failure_reinf,1000)
    strains_concrete=np.linspace(0,Strain_failure_concrete,1000)
    stress_concrete=Young_modulus_concrete*strains_concrete
    stress_reinf=Young_modulus_reinf*strains_reinf
    yielded = strains_reinf > Strain_yield_reinf
    stress_reinf[yielded] = Stress_yield_reinf + Post_yield_modulus_reinf * (
        strains_reinf[yielded] - Strain_yield_reinf
    )
    fig, ax = plt.subplots()
    ax.plot(strains_reinf, stress_reinf, '-k')
    ax.plot(strains_concrete, stress_concrete, '-k')
    ax.grid()
    ax.set_xlabel(r'$\epsilon$ [-]')
    ax.set_ylabel(r'$\sigma$ [MPa]')
    plt.show()

#section and bond
D_reinf=10                       #diameter of the reinforcement [mm]
Area_reinf=np.pi*D_reinf**2/4  #area of the reinforcement[mm2]
D_gross=50                       #diameter of the section [mm]
Area_gross=np.pi*D_gross**2/4  #area of the section [mm2]
Area_concrete=Area_gross-Area_reinf #area of the concrete [mm2]
Tau0=1                           #shear stress between reinforcement and steel before yielding of the steel [MPa]
Tau1=0.5                         #shear stress between reinforcement and steel after yielding of the steel [MPa]

#preliminary operations (1)
Slope0=D_reinf*Tau0*np.pi      #slope of the curve ForceOnTheSteel-x, before yielding of the steel [N/mm]
Slope1=D_reinf*Tau1*np.pi      #slope of the curve ForceOnTheSteel-x, after yielding of the steel [N/mm]
CrackSpacing=Stress_failure_concrete*Area_concrete/Slope0*2 #maximum crack spacing [mm]
Lambda=0.67
CrackSpacing=CrackSpacing*Lambda

#test -> is the steel still not yielded in this moment?
F_total_bonded=Stress_failure_concrete\
    *(Area_reinf*Young_modulus_reinf+Area_concrete*Young_modulus_concrete)\
    /Young_modulus_concrete*Lambda                       #total force on the section in the moment of incipient crack [N]
Stress_steel_InThatMoment=F_total_bonded/Area_reinf    #stress on the steel in that moment [MPa]
if Stress_steel_InThatMoment>Stress_yield_reinf:
    ValueError("Steel already yielded in the moment of cracking! -> to be implemented.")

#preliminary operations (2)
F_total_when_steel_starts_yielding=Stress_yield_reinf*Area_reinf
F_total_when_the_entire_section_is_yielded=F_total_when_steel_starts_yielding+Slope1*CrackSpacing/2
F_total_at_failure= Stress_failure_reinf*Area_reinf   
if F_total_at_failure<F_total_when_the_entire_section_is_yielded:
    ValueError("Failure occured before the entire steel length is yielded! -> maybe to be implemented?")



#section tool --> increase force to see how forces distributes along the bar
F=39000
F=30000

def CalculateCurve(F):
    F1=F
    X1=0
    Fs_bonded=F*(Area_reinf*Young_modulus_reinf)/(Area_reinf*Young_modulus_reinf+Area_concrete*Young_modulus_concrete)
    Fc_bonded=F*(Area_concrete*Young_modulus_concrete)/(Area_reinf*Young_modulus_reinf+Area_concrete*Young_modulus_concrete)
    if F<=F_total_bonded:
        F2=F1
        X2=0
        F3=Fs_bonded
        X3=Fc_bonded/Slope0
    elif F<=F_total_when_steel_starts_yielding:
        F2=F1
        X2=0
        F3=F-Area_concrete*Stress_failure_concrete*Lambda
        X3=CrackSpacing/2
    elif F<=F_total_when_the_entire_section_is_yielded:
        F2=F_total_when_steel_starts_yielding
        X2=(F-F_total_when_steel_starts_yielding)/Slope1
        F3=F2-Slope0*(CrackSpacing/2-X2)
        X3=CrackSpacing/2
    elif F<=F_total_at_failure:
        F2=F-CrackSpacing/2*Slope1
        X2=CrackSpacing/2
        F3=F-CrackSpacing/2*Slope1
        X3=CrackSpacing/2
    F4=F3
    F5=F2
    F6=F1
    X4=CrackSpacing-X3
    X5=CrackSpacing-X2
    X6=CrackSpacing-X1
    return X1,X2,X3,X4,X5,X6,F1,F2,F3,F4,F5,F6

X1,X2,X3,X4,X5,X6,F1,F2,F3,F4,F5,F6=CalculateCurve(F)

def GetEpsOnSteel(ForceOnSteel):
    Sigma=ForceOnSteel/Area_reinf
    if Sigma<=Stress_yield_reinf:
        Eps=Sigma/Young_modulus_reinf
    else:
        Eps=Strain_yield_reinf+(Sigma-Stress_yield_reinf)/Post_yield_modulus_reinf
    return Eps

def CalculateTotalStrain(F):
    X1,X2,X3,X4,X5,X6,F1,F2,F3,F4,F5,F6=CalculateCurve(F)
    Eps1=GetEpsOnSteel(F1)
    Eps2=GetEpsOnSteel(F2)
    Eps3=GetEpsOnSteel(F3)
    Eps4=GetEpsOnSteel(F4)
    Eps5=GetEpsOnSteel(F5)
    Eps6=GetEpsOnSteel(F6)
    L12=X2-X1
    L23=X3-X2
    L34=X4-X3
    L45=X5-X4
    L56=X6-X5
    Eps12=(Eps1+Eps2)/2
    Eps23=(Eps2+Eps3)/2
    Eps34=(Eps3+Eps4)/2
    Eps45=(Eps4+Eps5)/2
    Eps56=(Eps5+Eps6)/2
    Eps=(Eps12*L12+Eps23*L23+Eps34*L34+Eps45*L45+Eps56*L56)/(L12+L23+L34+L45+L56)
    return Eps

F_Curve=np.linspace(0,F_total_at_failure,1000)
Strain_Curve=np.array([CalculateTotalStrain(force) for force in F_Curve])

Strain_NakedSteel=np.array([GetEpsOnSteel(force) for force in F_Curve])

Force_Bonded=np.linspace(0,F_total_bonded,100)
Stiffness_bondedsection=1/(Area_concrete*Young_modulus_concrete+Area_reinf*Young_modulus_reinf)
Strain_Bonded=Force_Bonded*Stiffness_bondedsection

fig, (ax, strain_ax) = plt.subplots(1, 2, figsize=(12, 5))
line, = ax.plot([X1,X2,X3,X4,X5,X6,X1], [F1,F2,F3,F4,F5,F6,F1], '-k')
ax.axhline(y=F_total_bonded, linestyle='--')
ax.axhline(y=F_total_when_steel_starts_yielding, linestyle='--')
ax.axhline(y=F_total_when_the_entire_section_is_yielded, linestyle='--')
ax.axhline(y=F_total_at_failure, linestyle='--')
ax.grid()
ax.set_xlim(0,CrackSpacing)
ax.set_ylim(0,F_total_at_failure)

ax.set_xlabel(r'$\epsilon$ [-]')
ax.set_ylabel(r'$\sigma$ [MPa]')
strain_ax.plot(Strain_Curve, F_Curve, '-k')
strain_ax.plot(Strain_NakedSteel, F_Curve,linestyle='--')
strain_ax.plot(Strain_Bonded, Force_Bonded,linestyle='--')
force_indicator = strain_ax.axhline(y=F, color='r', linestyle='--')
strain_ax.grid()
strain_ax.set_ylim(0, F_total_at_failure)
strain_ax.set_xlabel(r'$\epsilon$ [-]')
strain_ax.set_ylabel('F [N]')
fig.subplots_adjust(right=0.88, wspace=0.35)
slider_ax = fig.add_axes([0.92, 0.15, 0.025, 0.7])
force_slider = Slider(
    slider_ax,
    'F',
    0,
    F_total_at_failure,
    valinit=F,
    orientation='vertical',
)

def UpdateCurve(force):
    curve = CalculateCurve(force)
    x_values = curve[:6]
    force_values = curve[6:]
    line.set_data(
        [*x_values, x_values[0]],
        [*force_values, force_values[0]],
    )
    force_indicator.set_ydata([force, force])
    fig.canvas.draw_idle()

force_slider.on_changed(UpdateCurve)

plt.show()







