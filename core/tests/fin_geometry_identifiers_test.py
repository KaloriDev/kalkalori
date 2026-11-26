"""Stable fin identifiers at enum, serialization and dispatch boundaries."""
import json
import pickle

import pytest

from core.geometry import TubeSurfaceType
from core.heat_transfer.outside_dispatch import tube_surface_type


@pytest.mark.parametrize("member,identifier", [
    (TubeSurfaceType.PLAIN, "plain"),
    (TubeSurfaceType.CIRCULAR_FINNED, "circular_fin"),
    (TubeSurfaceType.SERRATED_HELICAL_FIN, "serrated_helical_fin"),
    (TubeSurfaceType.RECTANGULAR_INDIVIDUAL_FINNED, "rectangular_fin"),
    (TubeSurfaceType.CONTINUOUS_PLATE_FIN, "continuous_plate_fin"),
])
def test_canonical_json_and_enum_names_round_trip(member, identifier):
    payload = json.loads(json.dumps({"surface_type": member}))
    assert payload == {"surface_type": identifier}
    assert TubeSurfaceType(payload["surface_type"]) is member
    assert TubeSurfaceType[member.name] is member


@pytest.mark.parametrize("legacy,member", [
    ("circular_finned", TubeSurfaceType.CIRCULAR_FINNED),
    ("rectangular_individual_finned", TubeSurfaceType.RECTANGULAR_INDIVIDUAL_FINNED),
])
def test_legacy_json_and_enum_pickles_emit_canonical_values(legacy, member):
    restored = TubeSurfaceType(json.loads(json.dumps(legacy)))
    assert restored is member
    assert json.loads(json.dumps(restored)) == member.value != legacy

    class LegacyEnumPickle:
        def __reduce__(self):
            # Enum pickles reconstruct the member by its historical string value.
            return TubeSurfaceType, (legacy,)

    restored = pickle.loads(pickle.dumps(LegacyEnumPickle()))
    assert restored is member
    assert member.value.encode() in pickle.dumps(restored)
    assert legacy.encode() not in pickle.dumps(restored)


@pytest.mark.parametrize("identifier", ["unknown_fin", "circular", "rectangular", None, []])
def test_aliases_do_not_widen_other_geometry_identifiers(identifier):
    with pytest.raises(ValueError):
        TubeSurfaceType(identifier)


@pytest.mark.parametrize("family,identifier", [
    ("circular", "circular_fin"),
    ("rectangular", "rectangular_fin"),
    ("serrated", "serrated_helical_fin"),
    ("continuous", "continuous_plate_fin"),
])
def test_real_geometry_dispatch_serializes_canonical_surface(family, identifier):
    from core.tests.circular_finned_tube_integration_test import _bundle
    from core.tests.zhao2017_fixtures import source_bundle
    from core.tests.serrated_finned_dispatch_test import bank
    from core.tests.continuous_fin_integration_test import geometry

    factories = {"circular": lambda: _bundle(finned=True), "rectangular": source_bundle,
                 "serrated": bank, "continuous": geometry}
    surface = tube_surface_type(factories[family]())
    assert json.loads(json.dumps(surface)) == identifier
