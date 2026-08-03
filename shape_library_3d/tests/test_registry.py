"""The curated registry binds each shape to a generator and its curation metadata."""
import shape_library_3d.generators as G
from shape_library_3d.registry import SHAPES, ShapeSpec, iter_specs


def test_registry_composition_by_tier():
    tiers = [s.tier for s in SHAPES.values()]
    assert tiers.count(1) == 10
    assert tiers.count(2) == 2
    assert tiers.count(3) == 2
    assert len(SHAPES) == 14


def test_every_spec_resolves_a_generator_and_curation():
    for name, spec in SHAPES.items():
        assert isinstance(spec, ShapeSpec)
        assert hasattr(G, spec.generator), f"{name}: missing generator {spec.generator}"
        assert spec.rf_characteristic.strip(), f"{name}: empty rf_characteristic"
        assert spec.numerical_characteristic.strip(), f"{name}: empty numerical_characteristic"
        assert spec.role in {"control", "stressor", "reject"}


def test_tiers_carry_target_volume_but_rejects_do_not():
    for spec in SHAPES.values():
        if spec.tier in (1, 2):
            assert spec.target_volume_mm3 is not None
        else:
            assert spec.target_volume_mm3 is None


def test_control_shape_ordered_first():
    assert list(SHAPES)[0] == "cube"  # flat orthogonal control leads (mirrors 2-D gate G3)


def test_iter_specs_matches_registry():
    assert list(iter_specs()) == list(SHAPES.items())
