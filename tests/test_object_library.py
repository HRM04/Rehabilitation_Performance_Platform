from src.platform_config.object_library import OBJECT_LIBRARY, get_object
from src.grasp.grasp_family import expected_families

LIBRARY = ["bottle", "cup", "spoon", "fork", "bowl", "book", "cell phone", "remote",
           "toothbrush", "scissors", "sports ball", "apple", "banana"]
GRASPS = {"cylindrical", "spherical", "large_palmar", "whole_hand_power", "handle", "hook",
          "pincer", "tip_pinch", "tripod", "lateral_pinch", "multidigit_precision"}


def test_all_thirteen_objects_present():
    assert sorted(OBJECT_LIBRARY) == sorted(LIBRARY)


def test_expected_grasps_known_and_unique():
    for name, o in OBJECT_LIBRARY.items():
        g = o["expected_grasps"]
        assert g and len(set(g)) == len(g) and set(g) <= GRASPS, name


def test_hybrid_changes_applied():
    assert get_object("bowl")["expected_grasps"] == ["large_palmar", "spherical"]
    assert get_object("sports ball")["expected_grasps"] == ["spherical", "large_palmar"]
    assert get_object("apple")["expected_grasps"] == ["spherical", "large_palmar"]
    assert get_object("bottle")["expected_grasps"] == ["cylindrical", "whole_hand_power"]


def test_only_cup_and_spoon_mix_both_families():
    mixed = {n for n, o in OBJECT_LIBRARY.items() if len(expected_families(o["expected_grasps"])) > 1}
    assert mixed == {"cup", "spoon"}
