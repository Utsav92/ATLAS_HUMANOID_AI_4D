"""Adapters for Spatial Intelligence & SLAM:
4. TUM RGB-D
5. EuRoC MAV
6. KITTI
7. TartanAir
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import numpy as np
from atlas.datasets.adapters.base import BaseDatasetAdapter
from atlas.datasets.schema import DatasetManifest, SLAMSample


class TUMRGBDAdapter(BaseDatasetAdapter):
    def _build_manifest(self) -> DatasetManifest:
        return DatasetManifest(
            name="TUM RGB-D",
            source_id="tum_rgbd_benchmark",
            version="1.0",
            license_spdx="CC-BY-3.0",
            robot_embodiment="Kinect RGB-D Sensor Rig",
            num_episodes=39,
            num_samples=2500,
            features=["rgb", "depth", "ground_truth_pose", "accelerometer"],
            coordinate_frame="RDF",
            citation="Sturm et al., A Benchmark for the Evaluation of RGB-D SLAM Systems, IROS 2012",
            url="https://cvg.cit.tum.de/data/datasets/rgbd-dataset",
            purpose="RGB-D SLAM and 3D reconstruction"
        )

    def get_sample_count(self) -> int:
        return 50

    def get_sample(self, index: int) -> SLAMSample:
        return SLAMSample(
            sequence_id="rgbd_dataset_freiburg1_xyz",
            frame_id=index,
            timestamp=1305031102.0 + index * 0.033,
            rgb_shape=(240, 320, 3),
            depth_shape=(240, 320),
            ground_truth_pose=np.array([index * 0.01, index * 0.005, 1.2, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            estimated_pose=np.array([index * 0.01 + 0.002, index * 0.005 - 0.001, 1.2, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        )


class EuRoCMAVAdapter(BaseDatasetAdapter):
    def _build_manifest(self) -> DatasetManifest:
        return DatasetManifest(
            name="EuRoC MAV",
            source_id="euroc_mav_dataset",
            version="1.0",
            license_spdx="CC-BY-4.0",
            robot_embodiment="Micro Aerial Vehicle Rig",
            num_episodes=11,
            num_samples=3000,
            features=["stereo_cam0", "stereo_cam1", "imu_angular_vel", "imu_linear_acc", "ground_truth_vicon"],
            coordinate_frame="RDF",
            citation="Burri et al., The EuRoC micro aerial vehicle datasets, IJRR 2016",
            url="https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets",
            purpose="Visual-inertial odometry and SLAM"
        )

    def get_sample_count(self) -> int:
        return 40

    def get_sample(self, index: int) -> SLAMSample:
        return SLAMSample(
            sequence_id="MH_01_easy",
            frame_id=index,
            timestamp=1403636579.0 + index * 0.05,
            rgb_shape=(240, 320, 1),
            depth_shape=(240, 320),
            ground_truth_pose=np.array([index * 0.02, 0.1, 0.8, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        )


class KITTIAdapter(BaseDatasetAdapter):
    def _build_manifest(self) -> DatasetManifest:
        return DatasetManifest(
            name="KITTI",
            source_id="kitti_odometry",
            version="2012",
            license_spdx="CC-BY-NC-SA-3.0",
            robot_embodiment="Autonomous Ground Vehicle",
            num_episodes=22,
            num_samples=4000,
            features=["stereo_color", "velodyne_lidar", "oxts_gps_imu", "ground_truth_poses"],
            coordinate_frame="FLU",
            citation="Geiger et al., Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite, CVPR 2012",
            url="https://www.cvlibs.net/datasets/kitti/eval_odometry.php",
            purpose="LiDAR and visual odometry"
        )

    def get_sample_count(self) -> int:
        return 50

    def get_sample(self, index: int) -> SLAMSample:
        return SLAMSample(
            sequence_id="kitti_seq_00",
            frame_id=index,
            timestamp=float(index) * 0.1,
            rgb_shape=(240, 320, 3),
            depth_shape=(240, 320),
            ground_truth_pose=np.array([index * 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        )


class TartanAirAdapter(BaseDatasetAdapter):
    def _build_manifest(self) -> DatasetManifest:
        return DatasetManifest(
            name="TartanAir",
            source_id="tartanair_synthetic",
            version="1.0",
            license_spdx="BSD-3-Clause",
            robot_embodiment="Simulated Quadrotor / Mobile Agent",
            num_episodes=100,
            num_samples=15000,
            features=["rgb_left", "depth_left", "optical_flow", "motion_seg", "camera_pose"],
            coordinate_frame="RDF",
            citation="Wang et al., TartanAir: A Dataset to Push the Limits of Visual SLAM, IROS 2020",
            url="https://tartanair.org/",
            purpose="Synthetic 3D vision, depth and SLAM"
        )

    def get_sample_count(self) -> int:
        return 60

    def get_sample(self, index: int) -> SLAMSample:
        return SLAMSample(
            sequence_id="abandonedfactory_Easy_P001",
            frame_id=index,
            timestamp=float(index) * 0.1,
            rgb_shape=(240, 320, 3),
            depth_shape=(240, 320),
            ground_truth_pose=np.array([index * 0.05, index * 0.02, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        )
