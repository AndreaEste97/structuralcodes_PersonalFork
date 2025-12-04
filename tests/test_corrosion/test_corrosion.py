import pytest

from structuralcodes.development.corrosion import calculate_velocity_of_corrosion,calculate_minimum_area_after_corrosion

@pytest.mark.parametrize(
    "corrosion_type, exposure_class, fractile, expected",
    [
        ("carbonation_induced", "Sheltered", 0.5, 2),
        ("carbonation_induced", "Unsheltered",0.5, 5),
        ("chloride_induced", "Wet",0.5, 4),
        ("chloride_induced", "Cyclic_dry_wet",0.5, 30),
        ("chloride_induced", "Airborn_seawater",0.5, 30),
        ("chloride_induced", "Submerged",0.5, 4),
        ("chloride_induced", "Tidal_zone",0.5, 50),
    ]
)
def test_calculate_velocity_of_corrosion_without_fractile(corrosion_type, exposure_class, fractile, expected):
    from structuralcodes import set_design_code
    set_design_code('ec2_2004')
    Pcorr_rep = calculate_velocity_of_corrosion(corrosion_type=corrosion_type,exposure_class=exposure_class)
    assert Pcorr_rep == expected

@pytest.mark.parametrize(
    "corrosion_type, exposure_class, fractile, expected",
    [
        ("carbonation_induced", "Sheltered", 0.8413, 2+3),
        ("carbonation_induced", "Unsheltered",0.8413, 5+1),
        ("chloride_induced", "Wet",0.8413, 4+6),
        ("chloride_induced", "Cyclic_dry_wet",0.8413, 30+40),
        ("chloride_induced", "Airborn_seawater",0.8413, 30+40),
        ("chloride_induced", "Submerged",0.8413, 4+7),
        ("chloride_induced", "Tidal_zone",0.8413, 50+100),
        ("carbonation_induced", "Sheltered", 0.5, 2),
        ("carbonation_induced", "Unsheltered",0.5, 5),
        ("chloride_induced", "Wet",0.5, 4),
        ("chloride_induced", "Cyclic_dry_wet",0.5, 30),
        ("chloride_induced", "Airborn_seawater",0.5, 30),
        ("chloride_induced", "Submerged",0.5, 4),
        ("chloride_induced", "Tidal_zone",0.5, 50),
    ]
)
def test_calculate_velocity_of_corrosion_with_fractile(corrosion_type, exposure_class, fractile, expected):
    from structuralcodes import set_design_code
    set_design_code('ec2_2004')
    Pcorr_rep = calculate_velocity_of_corrosion(corrosion_type=corrosion_type,exposure_class=exposure_class,fractile=fractile)
    assert abs(Pcorr_rep - expected) < expected*0.001 #<0.1% error

@pytest.mark.parametrize(
    "corrosion_type, exposure_class",
    [
        ("carbonation_induced", "Wet"),
        ("carbonation_induced", "Cyclic_dry_wet"),
        ("carbonation_induced", "Airborn_seawater"),
        ("carbonation_induced", "Submerged"),
        ("carbonation_induced", "Tidal_zone"),
        ("chloride_induced", "Sheltered"),
        ("chloride_induced", "chloride_induced")
    ]
)
def test_wrong_corrosion_type_and_exposure_class_combinations(corrosion_type, exposure_class):
    from structuralcodes import set_design_code
    with pytest.raises(Exception):
        set_design_code('ec2_2004')
        calculate_velocity_of_corrosion(corrosion_type=corrosion_type,exposure_class=exposure_class)

def test_no_design_code_or_wring_design_code():
    from structuralcodes import set_design_code
    with pytest.raises(Exception):
        calculate_velocity_of_corrosion(corrosion_type="carbonation_induced",exposure_class="Sheltered")
    with pytest.raises(Exception):
        set_design_code('invaliddesigncode')
        calculate_velocity_of_corrosion(corrosion_type="carbonation_induced",exposure_class="Sheltered")

def test_low_values_of_fractile():
    from structuralcodes import set_design_code
    with pytest.raises(Exception):
        set_design_code('ec2_2004')
        calculate_velocity_of_corrosion(corrosion_type="carbonation_induced",exposure_class="Sheltered",fractile=0.001)
