#todo: verificare se slip è come su STKO, aggiungere corrosione, semplificare il codice eccessivamente lungo

from structuralcodes.materials.reinforcement import ReinforcementMC2010
import numpy as np
import matplotlib.pyplot as plt
import sys
from structuralcodes.materials.constitutive_laws import create_constitutive_law



LD=15

# Create a constitutive law for the reinforcement
reinforcement = ReinforcementMC2010(fyk=450, Es=200000, ftk=530, epsuk=0.15,Delta=0.1, L_OVer_D_ratio=LD)
constitutive_law = create_constitutive_law(constitutive_law_name='AsdSteel1D', material=reinforcement)

# Plot stress-strain relationship
strains = np.linspace(-0.12,0.12,1000)
stresses = constitutive_law.get_stress(strains)
if False:
    fig, ax = plt.subplots()
    ax.plot(strains, stresses, '-k')
    ax.grid()
    ax.set_xlabel(r'$\epsilon$ [-]')
    ax.set_ylabel(r'$\sigma$ [MPa]')
    plt.show()
old_strains=strains
old_stresses=stresses

#Creazione legge Tau-slip manuale (tau-slip secondo tesi Caprili (MC90, che è diversa da MC2010))
def GenerateTauSlipRelation(slip1, slip2, slip3, alpha, fck):
    #slip1=fine tratto curvo
    #slip2=fine tratto piatto. In realtà la ho un po' cambiata per era coincidente con slip1 e così invece mostra un tratto piatto
    #slip3=fine tratto obliquo e inizio plateau indefinito
    #alpha=esponente primo tratto
    TauMax=1*np.sqrt(reinforcement._ftk) #!!!!!!!!in realtà intende fck. relazione empirica, devono essere in MPa
    TauMax=TauMax*6 #!!!!!!!!!!!!!!cose messe a caso per test. Lasciarla per vedere rottura prima di slittamento o commentarla per vedere slittamento prima di rottura
    TauPlateau=0.15 * TauMax
    slip=np.linspace(0,3,1000)
    tau=np.linspace(0,3,1000)
    tau[(slip >= 0) & (slip < slip1)] = TauMax * (slip[(slip >= 0) & (slip < slip1)] / slip1) ** alpha
    tau[(slip >= slip1) & (slip < slip2)] = TauMax
    tau[(slip >= slip2) & (slip < slip3)] = TauMax + (TauPlateau - TauMax) / (slip3 - slip2) * (slip[(slip >= slip2) & (slip < slip3)]-slip2)
    tau[slip >= slip3] = TauPlateau
    return slip,tau
slip,tau=GenerateTauSlipRelation(slip1=0.6, slip2=0.7, slip3=2.5, alpha=0.4, fck=reinforcement._ftk)
if False:
    fig, ax = plt.subplots()
    ax.plot(slip, tau, '-k')
    ax.grid()
    ax.set_xlabel('slip [?]')
    ax.set_ylabel('tau [MPa]')
    plt.show()

#Conversione della curva tau-slip in sigma_slip-espilon_slip
def ConvertTauSlipCurveToEpsSig(slip,tau,RebarDiameter, RebarAnchorageLength):
    #RebarDiameter [mm]
    #RebarAnchorageLength [m]
    RebarAnchorageLength=RebarAnchorageLength*1000 #convert to mm
    RebarAnchorageLength=RebarAnchorageLength/1000 #!!!!! fattore correttivo temporaneo che ci va altrimenti non torna nulla non so perchè
    Sigma=tau*(np.pi*RebarDiameter*RebarAnchorageLength)/(np.pi*RebarDiameter**2/4)
    RelativeSlip=slip/RebarAnchorageLength
    return Sigma, RelativeSlip
Sigma,RelativeSlip=ConvertTauSlipCurveToEpsSig(slip=slip,tau=tau,RebarDiameter=10, RebarAnchorageLength=10)
if False:
    fig, ax = plt.subplots()
    ax.plot(RelativeSlip, Sigma, '-k')
    ax.grid()
    ax.set_xlabel('RelativeSlip [-]')
    ax.set_ylabel('Sigma [MPa]')
    plt.show()

#robe vecchie per development
""" #modifica la curva SigmaEpsilon per tenere conto dello slip
AssumedSlip1 = slip[np.argmax(Sigma)]
RelativeSlip_OnlyFirstTrait=RelativeSlip[slip<=AssumedSlip1]
Sigma_OnlyFirstTrait=Sigma[slip<=AssumedSlip1]
MaxStressTensileCurve=np.max(stresses)
MaxStressSlipCurve=np.max(Sigma)
if MaxStressTensileCurve<=MaxStressSlipCurve:
    StressesModifiedCurve=stresses
    StrainModifiedCurve=strains+np.interp(StressesModifiedCurve,Sigma_OnlyFirstTrait,RelativeSlip_OnlyFirstTrait)
elif MaxStressTensileCurve>MaxStressSlipCurve:
    StressesModifiedCurve=Sigma
    StrainModifiedCurve=RelativeSlip+np.interp(StressesModifiedCurve,stresses,strains) """

# Shift each strain from the base curve by the corresponding first-branch slip.
strains = np.linspace(-0.12,0.12,1000)
stresses = constitutive_law.get_stress(strains)
first_slip_peak = int(np.argmax(Sigma))
first_slip_stress = Sigma[:first_slip_peak + 1]
first_slip_strain = RelativeSlip[:first_slip_peak + 1]
slip_strains = np.interp(
    np.abs(stresses),
    first_slip_stress,
    first_slip_strain,
)
bonded_strains = strains + np.sign(strains) * slip_strains
if True:
    fig, ax = plt.subplots()
    ax.plot(bonded_strains, stresses, '-k',linewidth=2)
    ax.plot(old_strains, old_stresses, '--k',linewidth=0.5)
    ax.plot(RelativeSlip, Sigma, '--r',linewidth=0.5)
    ax.plot(-RelativeSlip, -Sigma, '--r',linewidth=0.5)
    ax.legend(["Stress-Strain, assumed bond","Stress-Strain, perfect bond","equivalent tau-slip law"])
    ax.grid()
    ax.set_xlim(left=-0.2, right=0.2)
    #ax.set_ylim(bottom=-0.2, top=0.2)
    ax.set_xlabel(r'$\epsilon$ [-]')
    ax.set_ylabel(r'$\sigma$ [MPa]')
    plt.show()