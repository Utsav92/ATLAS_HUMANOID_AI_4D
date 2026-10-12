"""Unit tests for Milestone M2: Complete Dataset Hub & Canonical Schemas."""

import pytest
import numpy as np
from atlas.datasets.registry import DatasetRegistry
from atlas.datasets.adapters.base import BaseDatasetAdapter


def test_m2_all_14_datasets_registered():
    reg = DatasetRegistry()
    datasets = reg.list_datasets()
    assert len(datasets) == 14, f"Expected 14 datasets, got {len(datasets)}"
    
    # Verify expected names are present
    names = {d["name"] for d in datasets}
    expected = {
        "G1 Moves", "Unitree G1 Manipulation", "MuJoCo-generated trajectories",
        "TUM RGB-D", "EuRoC MAV", "KITTI", "TartanAir",
        "YCB-Sight", "Sparsh", "DROID", "Open X-Embodiment",
        "LeRobot", "Minari", "ATLAS Language-to-Action"
    }
    assert expected.issubset(names)


def test_m2_license_audit_and_integrity():
    reg = DatasetRegistry()
    licenses = reg.audit_licenses()
    assert len(licenses) == 14
    for src_id, lic in licenses.items():
        assert len(lic) > 0, f"License missing for {src_id}"

    integrity = reg.validate_all_integrities()
    for src_id, ok in integrity.items():
        assert ok is True, f"Integrity check failed for {src_id}"


def test_m2_coordinate_transformations():
    reg = DatasetRegistry()
    adapter = reg.get_adapter("tum_rgbd_benchmark")
    assert adapter is not None

    # Test RDF to FLU
    rdf_point = np.array([1.0, 2.0, 3.0], dtype=np.float32)  # X: right, Y: down, Z: forward
    flu_point = adapter.transform_coordinates(rdf_point, source_frame="RDF", target_frame="FLU")
    # Expected FLU: X: forward=3.0, Y: left=-1.0, Z: up=-2.0
    np.testing.assert_allclose(flu_point, np.array([3.0, -1.0, -2.0], dtype=np.float32))

    # Test round trip back to RDF
    rdf_roundtrip = adapter.transform_coordinates(flu_point, source_frame="FLU", target_frame="RDF")
    np.testing.assert_allclose(rdf_roundtrip, rdf_point)


def test_m2_joint_mapping_validation():
    reg = DatasetRegistry()
    adapter = reg.get_adapter("droid_manipulation")
    assert adapter is not None

    source_names = ["arm_joint_1", "arm_joint_2", "gripper"]
    source_vals = np.array([0.5, -0.3, 1.0], dtype=np.float32)

    target_g1_names = ["arm_joint_1", "arm_joint_2", "waist_yaw", "left_hip_pitch"]
    mapped_vals, mapped_names, missing_names = adapter.map_joints_to_g1(
        source_names, source_vals, target_g1_names
    )

    assert len(mapped_vals) == 4
    assert mapped_vals[0] == 0.5
    assert mapped_vals[1] == -0.3
    assert mapped_vals[2] == 0.0  # Missing joint defaulted to 0
    assert "waist_yaw" in missing_names
    assert "arm_joint_1" in mapped_names


def test_m2_train_val_test_splits():
    reg = DatasetRegistry()
    split = reg.get_train_val_test_split("atlas_lang_act_v1")
    assert "train" in split and "val" in split and "test" in split
    total = len(split["train"]) + len(split["val"]) + len(split["test"])
    adapter = reg.get_adapter("atlas_lang_act_v1")
    assert total == adapter.get_sample_count()

    # Verify no overlaps
    s_train, s_val, s_test = set(split["train"]), set(split["val"]), set(split["test"])
    assert len(s_train.intersection(s_val)) == 0
    assert len(s_train.intersection(s_test)) == 0
