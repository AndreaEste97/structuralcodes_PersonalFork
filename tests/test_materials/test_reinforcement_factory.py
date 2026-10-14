"""Tests for the reinforcement factory."""

import numpy as np
import pytest

from structuralcodes.codes import set_design_code
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
        -steel_stress, rel=1e-3
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
