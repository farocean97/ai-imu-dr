import os
import time
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.spatial.transform import Rotation as R

from utils_torch_filter import TORCHIEKF
from utils_numpy_filter import NUMPYIEKF as IEKF
from utils import prepare_data
from main_kitti import KITTIArgs
import argparse


def find_cutoff_timestamp(df, threshold_mph):
    """
    Finds the timestamp just before the first large jump in the trajectory.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing position data.
    - threshold_mph (float): The speed threshold in mph to determine a large jump.

    Returns:
    - int or None: The timestamp of the first large jump, or None if no jump is found.
    """
    x_col = 'pose_03'
    y_col = 'pose_13'
    time_col = 'timestamp_ns'

    # Calculate the Euclidean distance between consecutive points in meters
    distances_meters = np.sqrt(np.diff(df[x_col])**2 + np.diff(df[y_col])**2)

    # Conversion constants
    meters_to_miles = 1 / 1609.34
    nanoseconds_to_hours = 1e9 * 3600
    
    # Calculate speeds in mph
    distances_miles = distances_meters * meters_to_miles
    time_diffs_hours = np.diff(df[time_col]) / nanoseconds_to_hours
    speeds_mph = distances_miles / time_diffs_hours

    # Find the index of the first large jump in speed
    jump_index = np.where(speeds_mph > threshold_mph)[0]

    if len(jump_index) > 0:
        # Return the timestamp at the first large jump
        return df.iloc[jump_index[0]][time_col]
    else:
        return None


def get_data_paths(base_dir):
    base_path = Path(base_dir)
    data_paths = {}

    for folder in base_path.iterdir():
        if folder.is_dir():
            folder_name = folder.name

            # slam_pose = folder / 'twomap_ws' / 'graphs' / 'scan_matching' / 'step_1' / 'optimized' / 'vehicle_pose_nodes.parquet'
            # odometry = folder / 'twomap_ws' / 'sensor_data' / folder_name / 'odometry.parquet'

            imu = folder / f'{folder_name}_imu.parquet'
            wheel_speed = folder / f'{folder_name}_wheel_speed.parquet'
            calibration = folder / f'{folder_name}_calibration.json'
            interpolated_pose = folder / 'twomap_ws' / 'sensor_data' / folder_name / 'interpolated_pose.parquet'

            if imu.exists() and wheel_speed.exists() and calibration.exists() and interpolated_pose.exists():
                data_paths[folder_name] = {
                    'imu': str(imu),
                    'wheel_speed': str(wheel_speed),
                    'calibration': str(calibration),
                    'interpolated_pose': str(interpolated_pose),
                    'experience_name': folder_name,
                }

    return data_paths


def read_p4h_data(data_paths):
    """
    Reads the data from the specified paths.
    """
    imu_path = data_paths['imu']
    wheel_speed_path = data_paths['wheel_speed']
    calibration_path = data_paths['calibration']
    interpolated_pose_path = data_paths['interpolated_pose']

    imu_df = pd.read_parquet(imu_path)
    wheel_speed_df = pd.read_parquet(wheel_speed_path)
    interpolated_pose_df = pd.read_parquet(interpolated_pose_path)

    imu_df = imu_df.sort_values('timestamp_ns')
    wheel_speed_df = wheel_speed_df.sort_values('timestamp_ns')
    interpolated_pose_df = interpolated_pose_df.sort_values('timestamp_ns')

    with open(calibration_path, 'r') as file:
        calibration_json = json.load(file)

    threshold_mph = 100
    timestamp_cutoff = find_cutoff_timestamp(interpolated_pose_df, threshold_mph)

    if timestamp_cutoff is not None:
        # print(f"Large jump detected at timestamp: {timestamp_cutoff:.0f}")
        imu_df = imu_df[imu_df['timestamp_ns'] <= timestamp_cutoff]
        wheel_speed_df = wheel_speed_df[wheel_speed_df['timestamp_ns'] <= timestamp_cutoff]
        interpolated_pose_df = interpolated_pose_df[interpolated_pose_df['timestamp_ns'] <= timestamp_cutoff]

    return imu_df, wheel_speed_df, calibration_json, interpolated_pose_df


def prepare_p4h_data(data_paths):
    """
    Prepares the data for the filter.
    """
    # Read the data from the paths
    imu_df, wheel_speed_df, calibration_json, interpolated_pose_df = read_p4h_data(data_paths)

    # Processing IMU data
    imu_df = imu_df.iloc[::10].reset_index(drop=True) # Downsample IMU data
    # imu_df = imu_df.iloc[:20000]
    gyro_data = imu_df[['gyro_x_radps', 'gyro_y_radps', 'gyro_z_radps']].values
    acc_data = imu_df[['acc_x_mpss', 'acc_y_mpss', 'acc_z_mpss']].values
    u = np.concatenate((gyro_data, acc_data), axis=1)

    imu_timestamps_ns = imu_df['timestamp_ns'].values
    t = (imu_timestamps_ns - imu_timestamps_ns[0]) / 1e9  # Convert to seconds from start

    # Processing Wheel Speed data
    P4A_WHEEL_RADIUS = 0.36  # Define the wheel radius in meters
    wheel_speed_df['rear_left_speed_mps'] = wheel_speed_df['rear_left_speed_radps'] * P4A_WHEEL_RADIUS
    wheel_speed_df['rear_right_speed_mps'] = wheel_speed_df['rear_right_speed_radps'] * P4A_WHEEL_RADIUS
    wheel_speed_df['vehicle_mps'] = wheel_speed_df[['rear_left_speed_mps',
                                                       'rear_right_speed_mps']].mean(axis=1)

    vehicle_speed = wheel_speed_df['vehicle_mps'].values
    v = np.zeros((vehicle_speed.shape[0], 3))
    v[:, 0] = vehicle_speed  # x component
    # vehicle_speed_ts = wheel_speed_df['timestamp_ns'].values

    # Processing Interpolated Pose data
    # Create np.array T_EV
    T_EV_columns = [
        'pose_00', 'pose_01', 'pose_02', 'pose_03',
        'pose_10', 'pose_11', 'pose_12', 'pose_13',
        'pose_20', 'pose_21', 'pose_22', 'pose_23',
        'pose_30', 'pose_31', 'pose_32', 'pose_33'
    ]
    T_EV = interpolated_pose_df[T_EV_columns].values
    T_EV = T_EV.reshape((T_EV.shape[0], 4, 4))
    T_EV_ts = interpolated_pose_df['timestamp_ns'].values
    T_EV_ts = (T_EV_ts - T_EV_ts[0]) / 1e9  # Convert to seconds from start

    # Getting Car -> IMU transformation from calibration
    T_VI = np.eye(4)
    imu_pose = calibration_json["inertialNavSystems"][0]["pose"]
    T_VI[:3, :3] = np.array([
        imu_pose['rotation']['row1'],
        imu_pose['rotation']['row2'],
        imu_pose['rotation']['row3']
    ])
    T_VI[:3, 3] = np.array([
        imu_pose['translation']['x'],
        imu_pose['translation']['y'],
        imu_pose['translation']['z']
    ])

    T_VI_rpy = rot_to_rpy(T_VI[:3, :3])
    print(f"T_VI RPY (degrees): {[f'{x:.2f}' for x in T_VI_rpy]}")

    return t, u, v, T_EV_ts, T_EV, T_VI


class P4HParameters(IEKF.Parameters):
    # gravity vector
    g = np.array([0, 0, -9.80655])

    cov_omega = 2e-4
    cov_acc = 1e-3
    cov_b_omega = 1e-8
    cov_b_acc = 1e-6
    cov_Rot_c_i = 1e-8
    cov_t_c_i = 1e-8
    cov_Rot0 = 1e-6
    cov_v0 = 1e-1
    cov_b_omega0 = 1e-8
    cov_b_acc0 = 1e-3
    cov_Rot_c_i0 = 1e-5
    cov_t_c_i0 = 1e-2
    cov_lat = 1
    cov_up = 10

    def __init__(self, **kwargs):
        super(P4HParameters, self).__init__(**kwargs)
        self.set_param_attr()

    def set_param_attr(self):
        attr_list = [a for a in dir(P4HParameters) if
                     not a.startswith('__') and not callable(getattr(P4HParameters, a))]
        for attr in attr_list:
            setattr(self, attr, getattr(P4HParameters, attr))


def initialize_state(R_VI: np.ndarray, p_VI_V: np.ndarray, R_LV: np.ndarray, v_LV_Vx: float, w_LV: np.ndarray = np.array([0, 0, 0])) -> tuple:
    """
    Construct initial state given vehicle speed, IMU angular rate and IMU extrinsics.

    @param R_VI: IMU/vehicle frame rotation
    @param p_VI_V: IMU/vehicle frame translation
    @param R_LV: initial vehicle orientation wrt L frame
    @param v_LV_Vx: initial longitudinal vehicle speed
    
    @returns tuple[R_LI, p_LI_L, v_LI_L]
    """
    R_LI = R_LV @ R_VI
    p_LI_L = np.zeros(3)

    v_LV_V = np.array([v_LV_Vx, 0, 0])
    v_LI_L = R_LV @ (v_LV_V + np.cross(w_LV, p_VI_V))
    return R_LI, p_LI_L, v_LI_L


def rot_to_rpy(m):
    return R.from_matrix(m).as_euler("xyz", degrees=True)


def batch_rot_to_rpy(T_batch):
    """Convert a batch of transformation matrices to roll, pitch, yaw angles.
    Args:
        T_batch (np.ndarray): Array of shape (N, 4, 4)
    Returns:
        np.ndarray: Array of shape (N, 3) containing roll, pitch, yaw angles in degrees.
    """
    return np.array([rot_to_rpy(T[:3, :3]) for T in T_batch])


def compute_velocity_from_pose(pose: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    N = pose.shape[0]
    if N < 2:
        raise ValueError("Need at least two poses.")

    translations = pose[:, :3, 3]
    dt = np.diff(timestamps)
    d_pos = np.diff(translations, axis=0)

    dt = np.where(dt == 0, 1e-6, dt)  # prevent division by zero
    velocity = d_pos / dt[:, np.newaxis]
    velocity = np.vstack((np.zeros((1, 3)), velocity))

    return velocity


def plot_predicted_vs_ground_truth(pred_ts, pred_pose, gt_ts, gt_pose, experience_name):
    os.makedirs('./iekf_xyz_rpy_time_series/', exist_ok=True)
    pred_rpy = batch_rot_to_rpy(pred_pose)
    gt_rpy = batch_rot_to_rpy(gt_pose)

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # Row 1: X and Roll
    axes[0, 0].plot(gt_ts, gt_pose[:, 0, 3], 'ko', label="Twomap Interpolated Pose")
    axes[0, 0].plot(pred_ts, pred_pose[:, 0, 3], 'r.', label="IEKF Odometry")
    axes[0, 0].legend()
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('X')
    axes[0, 0].set_title('X coordinate - Time Series')
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(gt_ts, gt_rpy[:, 0], 'ko', label="Twomap Interpolated Pose")
    axes[0, 1].plot(pred_ts, pred_rpy[:, 0], 'r.', label="IEKF Odometry")
    axes[0, 1].legend()
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Roll (degrees)')
    axes[0, 1].set_title('Roll - Time Series')
    axes[0, 1].grid(True)
    
    # Row 2: Y and Pitch
    axes[1, 0].plot(gt_ts, gt_pose[:, 1, 3], 'ko', label="Twomap Interpolated Pose")
    axes[1, 0].plot(pred_ts, pred_pose[:, 1, 3], 'r.', label="IEKF Odometry")
    axes[1, 0].legend()
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Y')
    axes[1, 0].set_title('Y coordinate - Time Series')
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(gt_ts, gt_rpy[:, 1], 'ko', label="Twomap Interpolated Pose")
    axes[1, 1].plot(pred_ts, pred_rpy[:, 1], 'r.', label="IEKF Odometry")
    axes[1, 1].legend()
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Pitch (degrees)')
    axes[1, 1].set_title('Pitch - Time Series')
    axes[1, 1].grid(True)
    
    # Row 3: Z and Yaw
    axes[2, 0].plot(gt_ts, gt_pose[:, 2, 3], 'ko', label="Twomap Interpolated Pose")
    axes[2, 0].plot(pred_ts, pred_pose[:, 2, 3], 'r.', label="IEKF Odometry")
    axes[2, 0].legend()
    axes[2, 0].set_xlabel('Time')
    axes[2, 0].set_ylabel('Z')
    axes[2, 0].set_title('Z coordinate - Time Series')
    axes[2, 0].grid(True)
    
    axes[2, 1].plot(gt_ts, gt_rpy[:, 2], 'ko', label="Twomap Interpolated Pose")
    axes[2, 1].plot(pred_ts, pred_rpy[:, 2], 'r.', label="IEKF Odometry")
    axes[2, 1].legend()
    axes[2, 1].set_xlabel('Time')
    axes[2, 1].set_ylabel('Yaw (degrees)')
    axes[2, 1].set_title('Yaw - Time Series')
    axes[2, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'./iekf_xyz_rpy_time_series/{experience_name}_predicted_vs_ground_truth_pose.png')
    plt.close()


def plot_R_VI(T_VI_ts, T_VI, experience_name):
    os.makedirs('./iekf_T_VI/', exist_ok=True)
    rpy = batch_rot_to_rpy(T_VI)

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # Plot translation components
    axes[0, 0].plot(T_VI_ts, T_VI[:, 0, 3], 'b-', label="T_VI X")
    axes[0, 0].legend()
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('X (m)')
    axes[0, 0].set_title('T_VI X Translation - Time Series')
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(T_VI_ts, rpy[:, 0], 'b-', label="T_VI Roll")
    axes[0, 1].legend()
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Roll (degrees)')
    axes[0, 1].set_title('T_VI Roll - Time Series')
    axes[0, 1].grid(True)
    
    # Row 2: Y and Pitch
    axes[1, 0].plot(T_VI_ts, T_VI[:, 1, 3], 'b-', label="T_VI Y")
    axes[1, 0].legend()
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Y (m)')
    axes[1, 0].set_title('T_VI Y Translation - Time Series')
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(T_VI_ts, rpy[:, 1], 'b-', label="T_VI Pitch")
    axes[1, 1].legend()
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Pitch (degrees)')
    axes[1, 1].set_title('T_VI Pitch - Time Series')
    axes[1, 1].grid(True)
    
    # Row 3: Z and Yaw
    axes[2, 0].plot(T_VI_ts, T_VI[:, 2, 3], 'b-', label="T_VI Z")
    axes[2, 0].legend()
    axes[2, 0].set_xlabel('Time')
    axes[2, 0].set_ylabel('Z (m)')
    axes[2, 0].set_title('T_VI Z Translation - Time Series')
    axes[2, 0].grid(True)
    
    axes[2, 1].plot(T_VI_ts, rpy[:, 2], 'b-', label="T_VI Yaw")
    axes[2, 1].legend()
    axes[2, 1].set_xlabel('Time')
    axes[2, 1].set_ylabel('Yaw (degrees)')
    axes[2, 1].set_title('T_VI Yaw - Time Series')
    axes[2, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'./iekf_T_VI/{experience_name}_T_VI_time_series.png')
    plt.close()


def plot_local_frame_velocity(pred_ts, v_pred, gt_ts, v_gt, experience_name):
    os.makedirs('./iekf_local_frame_velocity/', exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    axes[0].plot(gt_ts, v_gt[:, 0], 'ko', label="Twomap Interpolated Pose")
    axes[0].plot(pred_ts, v_pred[:, 0], 'r.', label="IEKF Odometry")
    axes[0].legend()
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Vx (m/s)')
    axes[0].set_title('Vx - Time Series')
    axes[0].grid(True)

    axes[1].plot(gt_ts, v_gt[:, 1], 'ko', label="Twomap Interpolated Pose")
    axes[1].plot(pred_ts, v_pred[:, 1], 'r.', label="IEKF Odometry")
    axes[1].legend()
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Vy (m/s)')
    axes[1].set_title('Vy - Time Series')
    axes[1].grid(True)

    axes[2].plot(gt_ts, v_gt[:, 2], 'ko', label="Twomap Interpolated Pose")
    axes[2].plot(pred_ts, v_pred[:, 2], 'r.', label="IEKF Odometry")
    axes[2].legend()
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Vz (m/s)')
    axes[2].set_title('Vz - Time Series')
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(f'./iekf_local_frame_velocity/{experience_name}_predicted_vs_ground_truth_velocity.png')
    plt.close()


def plot_vehicle_frame_velocity(v, t, experience_name):
    os.makedirs('./iekf_vehicle_frame_velocity/', exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(10, 16))

    axes[0].plot(t, v[:, 0], 'b-', label="Vehicle Vx")
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Vx (m/s)')
    axes[0].set_title('Vehicle Frame Velocity Vx - Time Series')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t, v[:, 1], 'g-', label="Vehicle Vy")
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Vy (m/s)')
    axes[1].set_title('Vehicle Frame Velocity Vy - Time Series')
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(t, v[:, 2], 'r-', label="Vehicle Vz")
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Vz (m/s)')
    axes[2].set_title('Vehicle Frame Velocity Vz - Time Series')
    axes[2].legend()
    axes[2].grid(True)

    # Add a 4th subplot that shows all velocity components together
    axes[3] = fig.add_subplot(4, 1, 4)
    axes[3].plot(t, v[:, 1], 'g-', label="Vehicle Vy", linewidth=2)
    axes[3].plot(t, v[:, 2], 'r-', label="Vehicle Vz", linewidth=2)
    axes[3].set_xlabel('Time (s)')
    axes[3].set_ylabel('Velocity (m/s)')
    axes[3].set_title('Vehicle Frame Vy and Vz - Combined')
    axes[3].legend()
    axes[3].grid(True)

    plt.tight_layout()
    plt.savefig(f'./iekf_vehicle_frame_velocity/{experience_name}_vehicle_frame_velocity.png')
    plt.close()


def plot_imu_frame_velocity(v, t, experience_name):
    os.makedirs('./iekf_imu_frame_velocity/', exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(10, 16))

    axes[0].plot(t, v[:, 0], 'b-', label="IMU Vx")
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Vx (m/s)')
    axes[0].set_title('IMU Frame Velocity Vx - Time Series')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t, v[:, 1], 'g-', label="IMU Vy")
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Vy (m/s)')
    axes[1].set_title('IMU Frame Velocity Vy - Time Series')
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(t, v[:, 2], 'r-', label="IMU Vz")
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Vz (m/s)')
    axes[2].set_title('IMU Frame Velocity Vz - Time Series')
    axes[2].legend()
    axes[2].grid(True)

    # Add a 4th subplot that shows all velocity components together
    axes[3] = fig.add_subplot(4, 1, 4)
    axes[3].plot(t, v[:, 0], 'b-', label="IMU Vx", linewidth=2)
    axes[3].plot(t, v[:, 1], 'g-', label="IMU Vy", linewidth=2)
    axes[3].plot(t, v[:, 2], 'r-', label="IMU Vz", linewidth=2)
    axes[3].set_xlabel('Time (s)')
    axes[3].set_ylabel('Velocity (m/s)')
    axes[3].set_title('IMU Frame Velocity Components - Combined')
    # axes[3].set_title('IMU Frame Vy and Vz - Combined')
    axes[3].legend()
    axes[3].grid(True)

    plt.tight_layout()
    plt.savefig(f'./iekf_imu_frame_velocity/{experience_name}_IMU_frame_velocity.png')
    plt.close()


def test_filter(args, dataset, data_paths):
    iekf = IEKF()
    torch_iekf = TORCHIEKF()

    # put Kitti parameters
    iekf.filter_parameters = P4HParameters()
    iekf.set_param_attr()
    torch_iekf.filter_parameters = P4HParameters()
    torch_iekf.set_param_attr()

    torch_iekf.load(args, dataset)
    iekf.set_learned_covariance(torch_iekf)

    t, u, v, T_EV_ts, T_EV, T_VI = prepare_p4h_data(data_paths)
    N, p = t.shape[0], None
    measurements_covs = np.tile(np.array([iekf.cov_lat, iekf.cov_up], dtype=np.float64), (N, 1))
    ang = np.zeros(3)  # Assuming Identity orientation

    R_LV = np.eye(3)
    R_LI, p_LI_L, v_LI_L = initialize_state(np.linalg.inv(T_VI[:3, :3]), T_VI[:3, 3], R_LV, v[0, 0], u[0, :3])
    p4h_params = {'T_VI': T_VI,
                  'R_LI': R_LI,
                  'p_LI_L': p_LI_L,
                  'v_LI_L': v_LI_L}

    print(f"Running IEKF for experience: {data_paths['experience_name']}")
    start_time = time.time()
    Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i = iekf.run(t, u, measurements_covs, v, p, N, ang, p4h_params)
    print(f"Execution time: {time.time() - start_time} seconds")

    # Transforming T_EV to start from identity
    T_EV = np.linalg.inv(T_EV[0]) @ T_EV

    os.makedirs('./iekf_xy_trajectories/', exist_ok=True)

    # Plotting TwoMap interpolated trajectory and IEKF trajectory
    fig, ax0 = plt.subplots(1, 1, figsize=(5, 5))
    ax0.plot(p[:, 0], p[:, 1], label="IEKF")
    ax0.plot(T_EV[:, 0, 3], T_EV[:, 1, 3], label="TwoMap interpolated pose")
    ax0.legend()
    ax0.set_xlabel('X')
    ax0.set_ylabel('Y')
    ax0.set_title(f"{data_paths['experience_name']} XY Trajectory \n n_secs: {t[-1]}", fontsize=10)
    ax0.grid(True)
    ax0.axis('equal')
    plt.savefig(f"./iekf_xy_trajectories/{data_paths['experience_name']}_XY_trajectories.png")
    plt.close()

   # Create T_pred transformation matrix from Rot and p
    T_pred = np.zeros((Rot.shape[0], 4, 4))
    T_pred[:, :3, :3] = Rot
    T_pred[:, :3, 3] = p
    T_pred[:, 3, 3] = 1
    plot_predicted_vs_ground_truth(t, T_pred, T_EV_ts, T_EV, data_paths['experience_name'])

    v_pred = compute_velocity_from_pose(T_pred, t)
    v_gt = compute_velocity_from_pose(T_EV, T_EV_ts)
    plot_local_frame_velocity(t, v_pred, T_EV_ts, v_gt, data_paths['experience_name'])
    
    v_imu = np.einsum('ijk,ik->ij', Rot.transpose(0, 2, 1), v)
    v_veh = np.einsum('ijk,ik->ij', Rot_c_i.transpose(0, 2, 1), v_imu)
    # v_imu = np.zeros_like(v)
    # v_veh = np.zeros_like(v)
    # for i in range(len(v)):
    #     v_imu[i] = Rot[i].T.dot(v[i])
    #     v_veh[i] = Rot_c_i[i].T.dot(v_imu[i])
    plot_vehicle_frame_velocity(v_veh, t, data_paths['experience_name'])
    plot_imu_frame_velocity(v_imu, t, data_paths['experience_name'])

    T_VI = np.zeros((Rot_c_i.shape[0], 4, 4))
    T_VI[:, :3, :3] = Rot_c_i
    T_VI[:, :3, 3] = t_c_i
    T_VI[:, 3, 3] = 1
    plot_R_VI(t, T_VI, data_paths['experience_name'])


if __name__ == "__main__":
    args = KITTIArgs()
    dataset = args.dataset_class(args)

    os.makedirs(args.path_results, exist_ok=True)
    print("Results will be saved in: " + args.path_results)
    parser = argparse.ArgumentParser(description='Run IEKF on P4H data')
    parser.add_argument('data_base_dir', type=Path, help='Base directory containing the data folders')
    
    parsed_args = parser.parse_args()
    data_base_dir = parsed_args.data_base_dir.expanduser()
    data_paths = get_data_paths(data_base_dir)

    for folder_name, data_paths in data_paths.items():
        test_filter(args, dataset, data_paths)
