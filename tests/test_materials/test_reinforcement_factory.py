"""Tests for the reinforcement factory."""

import numpy as np
import pytest

from structuralcodes.codes import set_design_code
from structuralcodes.materials.constitutive_laws import (
    AsdSteel1D,
    create_constitutive_law,
)
from structuralcodes.materials.reinforcement import (
    ReinforcementEC2_2004,
    ReinforcementEC2_2023,
    ReinforcementMC2010,
    create_reinforcement,
)


@pytest.mark.parametrize(
    'design_code, expected_reinforcement',
    [
        (None, None),
        ('mc2010', ReinforcementMC2010),
        ('ec2_2004', ReinforcementEC2_2004),
        ('ec2_2023', ReinforcementEC2_2023),
    ],
)
def test_reinforcement_factory(design_code, expected_reinforcement):
    """Test the reinforcement factory."""
    # Arrange
    fyk = 500
    Es = 200000
    ftk = 1.15 * fyk
    epsuk = 7.5e-2

    # Act and assert
    if design_code is not None:
        set_design_code(design_code=design_code)
        reinforcement = create_reinforcement(
            fyk=fyk, Es=Es, ftk=ftk, epsuk=epsuk
        )
        assert isinstance(reinforcement, expected_reinforcement)
    else:
        set_design_code(design_code=design_code)
        with pytest.raises(ValueError):
            create_reinforcement(fyk=fyk, Es=Es, ftk=ftk, epsuk=epsuk)


def test_mc2010_reinforcement_supports_asd_steel_without_eh():
    """AsdSteel1D derives its hardening modulus from reinforcement properties."""
    reinforcement = ReinforcementMC2010(
        fyk=500,
        Es=200000,
        ftk=510,
        epsuk=0.06,
        constitutive_law='AsdSteel1D',
    )

    law = reinforcement.constitutive_law
    expected_eh = (reinforcement.ftd() - reinforcement.fyd()) / (
        reinforcement.epsud() - reinforcement.fyd() / reinforcement.Es
    )

    assert law.__class__.__name__ == 'AsdSteel1D'
    assert law._Eh == pytest.approx(expected_eh)


@pytest.mark.parametrize(
    'stress_slip, strain_slip, stress_limit, steel_strain',
    [
        (
            [0, 100, 200, 250, 200],
            [0, 0.001, 0.004, 0.01, 0.03],
            250,
            0.0012,
        ),
        ([0, 800], [0, 0.02], 800, 0.02),
    ],
)
def test_mc2010_asd_steel_combines_stress_slip_curve(
    stress_slip, strain_slip, stress_limit, steel_strain
):
    """The factory should create an AsdSteel1D law with the slip curve."""
    reinforcement = ReinforcementMC2010(
        fyk=450,
        Es=200000,
        ftk=530,
        epsuk=0.15,
        constitutive_law='AsdSteel1D',
        Stress_Slip=np.asarray(stress_slip),
        Strain_Slip=np.asarray(strain_slip),
    )

    law = reinforcement.constitutive_law
    assert law._bonded_strain is not None
    assert np.max(law._bonded_stress) == pytest.approx(
        min(stress_limit, law._get_base_stress(law._eps_su))
    )

    steel_stress = law._get_base_stress(steel_strain)
    if stress_limit == 800:
        slip_stress = stress_slip[:2]
        slip_strains = strain_slip[:2]
    else:
        slip_stress = stress_slip[:4]
        slip_strains = strain_slip[:4]
    slip_strain = np.interp(steel_stress, slip_stress, slip_strains)
    total_strain = steel_strain + slip_strain
    assert law.get_stress(total_strain) == pytest.approx(
        steel_stress, rel=1e-3
    )
    assert law.get_stress(-total_strain) == pytest.approx(
        -steel_stress, rel=2e-3
    )
    if stress_limit == 250:
        assert law.get_stress(law._bonded_strain[-1] + 0.01) == pytest.approx(
            stress_slip[-1]
        )


def test_asd_steel_requires_stress_and_strain_slip_together():
    """Stress and strain slip data must be supplied as a pair."""
    with pytest.raises(ValueError, match='must be provided together'):
        ReinforcementMC2010(
            fyk=450,
            Es=200000,
            ftk=530,
            epsuk=0.15,
            constitutive_law='AsdSteel1D',
            Stress_Slip=[0, 100],
        )


@pytest.mark.parametrize(
    'ratio, compressive_strains, expected_stresses',
    [
        (
            None,
            [-0.001, -0.003],
            [-181.21948342418094, -422.10255920708147],
        ),
        (
            5.9,
            [-0.001, -0.003],
            [-181.21948342418094, -422.10255920708147],
        ),
        (6, [-0.001, -0.003], [-200, -450]),
        (8, [-0.001, -0.003], [-200, -450]),
        (8.1, [-0.001, -0.003], [-200, -448.7880063525123]),
    ],
)
def test_asd_steel_compression_curve_supports_scalars_and_arrays(
    ratio, compressive_strains, expected_stresses
):
    """Compression behavior should be defined for each slenderness range."""
    law = AsdSteel1D(
        E=200000,
        fy=450,
        Delta=0.1,
        eps_su=0.15,
        ftk=530,
        L_OVer_D_ratio=ratio,
    )

    actual_array = law._get_base_stress(np.asarray(compressive_strains))
    assert actual_array == pytest.approx(expected_stresses)
    assert law._get_base_stress(compressive_strains[0]) == pytest.approx(
        expected_stresses[0]
    )


def test_mc2010_passes_l_over_d_ratio_to_asd_steel_factory():
    """The material factory should forward the buckling ratio."""
    reinforcement = ReinforcementMC2010(
        fyk=450,
        Es=200000,
        ftk=530,
        epsuk=0.15,
        L_OVer_D_ratio=10,
    )

    law = create_constitutive_law('AsdSteel1D', reinforcement)
    reference_law = AsdSteel1D(
        E=reinforcement.Es,
        fy=reinforcement.fyd(),
        Delta=0.1,
        eps_su=reinforcement.epsud(),
        ftk=reinforcement.ftd(),
    )

    assert law._LDRatio == 10
    assert law.get_stress(-0.01) > reference_law.get_stress(-0.01)


@pytest.mark.parametrize('ratio', [0, -1, np.inf, np.nan])
def test_asd_steel_rejects_invalid_l_over_d_ratio(ratio):
    """Buckling ratio must be finite and positive."""
    with pytest.raises(ValueError, match='finite positive'):
        ReinforcementMC2010(
            fyk=450,
            Es=200000,
            ftk=530,
            epsuk=0.15,
            constitutive_law='AsdSteel1D',
            L_OVer_D_ratio=ratio,
        )


def test_asd_steel_slip_composition_uses_compression_curve():
    """Compression slip strain should be matched against steel compression."""
    reinforcement = ReinforcementMC2010(
        fyk=450,
        Es=200000,
        ftk=530,
        epsuk=0.15,
        constitutive_law='AsdSteel1D',
        L_OVer_D_ratio=15,
        Stress_Slip=[0, 100, 300, 500, 350, 100],
        Strain_Slip=[0, 0.001, 0.005, 0.02, 0.04, 0.1],
    )
    law = reinforcement.constitutive_law

    compression_strains = np.linspace(0, law._eps_su, 1000)
    compression_stresses = np.abs(
        law._get_base_stress(-compression_strains)
    )
    peak_index = int(np.argmax(compression_stresses))
    steel_strain = np.interp(
        300,
        compression_stresses[: peak_index + 1],
        compression_strains[: peak_index + 1],
    )
    slip_strain = np.interp(
        300,
        [0, 100, 300],
        [0, 0.001, 0.005],
    )
    assert law.get_stress(-(steel_strain + slip_strain)) == pytest.approx(
        -300,
        rel=1e-2,
    )
    assert np.all(np.diff(law._bonded_compression_strain) > 0)

    steel_strain = 0.05
    steel_stress = abs(law._get_base_stress(-steel_strain))
    slip_strain = np.interp(
        steel_stress,
        [100, 350, 500],
        [0.1, 0.04, 0.02],
    )
    assert law.get_stress(-(steel_strain + slip_strain)) == pytest.approx(
        -steel_stress,
        rel=2e-2,
    )

    steel_strain = 0.12
    steel_stress = abs(law._get_base_stress(-steel_strain))
    slip_strain = np.interp(
        steel_stress,
        [100, 350, 500],
        [0.1, 0.04, 0.02],
    )
    assert law.get_stress(-(steel_strain + slip_strain)) == pytest.approx(
        -steel_stress,
        rel=2e-3,
    )

    compressive_strain = np.linspace(
        0, law._eps_su + max(law._bonded_compression_strain), 1000
    )
    compressive_stress = np.abs(law.get_stress(-compressive_strain))
    peak_index = int(np.argmax(compressive_stress))

    assert compressive_stress[peak_index] == pytest.approx(
        law._fy,
        rel=2e-2,
    )
    assert np.all(np.diff(compressive_stress[: peak_index + 1]) >= -1e-8)
    assert np.all(np.diff(compressive_stress[peak_index:]) <= 1e-8)
