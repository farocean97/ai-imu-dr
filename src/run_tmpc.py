import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import argparse
import sys
from pathlib import Path
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os

def get_schema(channel_name):
    """
    Get the appropriate PyArrow schema for each channel type.
    
    Args:
        channel_name: Name of the channel
    
    Returns:
        pa.Schema: PyArrow schema for the channel, or None for default conversion
    """
    if channel_name == 'all_speed_ego_motion_perception_inputs':
        return pa.schema([
            ('timestamp', pa.int64()),
            ('validity', pa.int32()),
            
            # ECT1G01 fields
            ('ect1g01_timestamp', pa.float64()),
            ('ect1g01_b_p', pa.int32()),
            ('ect1g01_b_r', pa.int32()),
            ('ect1g01_b_n', pa.int32()),
            ('ect1g01_b_d', pa.int32()),
            ('ect1g01_b_b', pa.int32()),
            
            # VSC1F01 fields
            ('vsc1f01_timestamp', pa.int32()),
            ('vsc1f01_vxfr', pa.int32()),
            ('vsc1f01_vxfl', pa.int32()),
            ('vsc1f01_vxrr', pa.int32()),
            ('vsc1f01_vxrl', pa.int32()),
            ('vsc1f01_vxfref', pa.int32()),
            ('vsc1f01_vxflef', pa.int32()),
            ('vsc1f01_vxrref', pa.int32()),
            ('vsc1f01_vxrlef', pa.int32()),
            
            # VSC1G30 fields
            ('vsc1g30_timestamp', pa.int32()),
            ('vsc1g30_xstop', pa.int32()),
            
            # VSC1G14 fields
            ('vsc1g14_timestamp', pa.int32()),
            ('vsc1g14_vwpsumfr', pa.int32()),
            ('vsc1g14_vwpsumfl', pa.int32()),
            ('vsc1g14_vwpsumrr', pa.int32()),
            ('vsc1g14_vwpsumrl', pa.int32()),
            ('vsc1g14_vxfrhds', pa.int32()),
            ('vsc1g14_vxflhds', pa.int32()),
            ('vsc1g14_vxrrhds', pa.int32()),
            ('vsc1g14_vxrlhds', pa.int32()),
            ('vsc1g14_vxfrf', pa.int32()),
            ('vsc1g14_vxflf', pa.int32()),
            ('vsc1g14_vxrrf', pa.int32()),
            ('vsc1g14_vxrlf', pa.int32()),
            ('vsc1g14_vwpfrpm', pa.int32()),
            ('vsc1g14_vwpflpm', pa.int32()),
            ('vsc1g14_vwprrpm', pa.int32()),
            ('vsc1g14_vwprlpm', pa.int32()),
            ('vsc1g14_vwprrpms', pa.int32()),
            ('vsc1g14_vwprlpms', pa.int32()),
            ('vsc1g14_vwpfrpms', pa.int32()),
            ('vsc1g14_vwpflpms', pa.int32()),
            ('vsc1g14_rejufext', pa.int32()),
            
            # VSC1G12 fields
            ('vsc1g12_timestamp', pa.int32()),
            ('vsc1g12_gx0', pa.int32()),
            ('vsc1g12_gy0', pa.int32()),
            ('vsc1g12_yaw0', pa.int32()),
            ('vsc1g12_gxiv', pa.int32()),
            ('vsc1g12_gxf', pa.int32()),
            ('vsc1g12_gxhr0', pa.int32()),
            ('vsc1g12_gyiv', pa.int32()),
            ('vsc1g12_gyf', pa.int32()),
            ('vsc1g12_gyhr0', pa.int32()),
            ('vsc1g12_yaws', pa.int32()),
            ('vsc1g12_yawf', pa.int32()),
            ('vsc1g12_yr1hr0', pa.int32()),
            
            # IMU1G01 fields
            ('imu_signal_timestamp', pa.int32()),
            ('imu_signal_crc_037', pa.int32()),
            ('imu_signal_cnt_037', pa.int32()),
            ('imu_signal_yaw', pa.float64()),   # After conversion
            ('imu_signal_roll', pa.float64()),  # After conversion
            ('imu_signal_pitch', pa.float64()), # After conversion
            ('imu_signal_gx', pa.float64()),
            ('imu_signal_gy', pa.float64()),
            ('imu_signal_gz', pa.float64()),
            ('imu_signal_temp', pa.int32()),
            ('imu_signal_roll_fail', pa.int32()),
            ('imu_signal_roll_inv', pa.int32()),
            ('imu_signal_pitch_fail', pa.int32()),
            ('imu_signal_pitch_inv', pa.int32()),
            ('imu_signal_yaw_fail', pa.int32()),
            ('imu_signal_yaw_inv', pa.int32()),
            ('imu_signal_gx_fail', pa.int32()),
            ('imu_signal_gx_inv', pa.int32()),
            ('imu_signal_gy_fail', pa.int32()),
            ('imu_signal_gy_inv', pa.int32()),
            ('imu_signal_gz_fail', pa.int32()),
            ('imu_signal_gz_inv', pa.int32()),
            ('imu_signal_temp_fail', pa.int32()),
            ('imu_signal_temp_inv', pa.int32()),
            
            # Converted wheel speeds
            ('fl_e_fr_whlspd', pa.float64()),
            ('fl_e_fl_whlspd', pa.float64()),
            ('fl_e_rr_whlspd', pa.float64()),
            ('fl_e_rl_whlspd', pa.float64()),
            
            # Log metadata
            ('log_index', pa.int64()),
            ('log_timestamp', pa.float64())
        ])
    
    elif channel_name == 'ublox_all_speed_egomotion_perception_inputs':
        # Define the satellite info struct schema
        sat_info_struct = pa.struct([
            ('gnss_id', pa.int32()),
            ('sv_id', pa.int32()),
            ('cno', pa.int32()),
            ('elev', pa.int32()),
            ('azim', pa.int32()),
            ('pr_res', pa.int32()),
            ('flags', pa.int32())
        ])
        
        return pa.schema([
            ('timestamp', pa.int64()),
            ('validity', pa.int32()),
            
            # UbloxNavPvat fields
            ('pvat_gps_time_ns', pa.int64()),
            ('pvat_fix_type', pa.int32()),
            ('pvat_lon', pa.float64()),
            ('pvat_lat', pa.float64()),
            ('pvat_height', pa.int32()),
            ('pvat_err_ellipse_major', pa.int32()),
            ('pvat_err_ellipse_minor', pa.int32()),
            ('pvat_mot_heading', pa.float64()),
            ('pvat_vel_n', pa.int32()),
            ('pvat_vel_e', pa.int32()),
            ('pvat_vel_d', pa.int32()),
            
            # UbloxNavDop fields
            ('dop_p_dop', pa.float64()),
            ('dop_v_dop', pa.float64()),
            ('dop_h_dop', pa.float64()),
            
            # UbloxNavSat fields
            ('sat_num_svs', pa.int32()),
            ('sat_sv_info', pa.list_(sat_info_struct)),
            
            # Log metadata
            ('log_index', pa.int64()),
            ('log_timestamp', pa.float64())
        ])
    elif channel_name == 'vehicle_pose':
        return pa.schema([
            ('timestamp', pa.int64()),
            ('validity', pa.int32()),
            ('local_frame_id', pa.string()),
            ('distance_traveled_cumulative', pa.float64()),
            ('distance_traveled_local_frame', pa.float64()),
            ('position_x', pa.float64()),
            ('position_y', pa.float64()),
            ('position_z', pa.float64()),
            ('orientation_w', pa.float64()),
            ('orientation_x', pa.float64()),
            ('orientation_y', pa.float64()),
            ('orientation_z', pa.float64()),
            ('velocity_L_x', pa.float64()),
            ('velocity_L_y', pa.float64()),
            ('velocity_L_z', pa.float64()),
            ('rotational_velocity_V_x', pa.float64()),
            ('rotational_velocity_V_y', pa.float64()),
            ('rotational_velocity_V_z', pa.float64()),
            ('acceleration_V_x', pa.float64()),
            ('acceleration_V_y', pa.float64()),
            ('acceleration_V_z', pa.float64()),
            ('provider', pa.int32()),
            ('has_fallback_request', pa.bool_()),
            ('log_index', pa.int64()),
            ('log_timestamp', pa.float64())
        ])
    
    # Return None for channels that don't need custom schema
    raise ValueError(f"Unknown channel name: {channel_name}")


def read_parquet_file(file_path, channel_name):
    """
    Read a parquet file with the specified schema.
    
    Args:
        file_path: Path to the parquet file
        channel_name: Channel name to determine the schema
        
    Returns:
        pandas.DataFrame: Loaded data
    """
    try:
        # Get the expected schema
        expected_schema = get_schema(channel_name)
        
        # Read the parquet file
        table = pq.read_table(file_path, schema=expected_schema)
        
        # Convert to pandas DataFrame
        df = table.to_pandas()
        
        print(f"Successfully loaded {file_path}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Data types:\n{df.dtypes}")
        print("="*50)
        
        return df
        
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        return None

def extract_ego_motion_data(ego_motion_df):
    """
    Extract relevant data from ego motion DataFrame.
    
    Args:
        ego_motion_df: DataFrame containing ego motion perception inputs
        
    Returns:
        pandas.DataFrame: Extracted data with columns:
            - timestamp: System timestamp
            - imu_gx: IMU gyroscope X-axis
            - imu_gy: IMU gyroscope Y-axis  
            - imu_gz: IMU gyroscope Z-axis
            - imu_roll: IMU roll angle
            - imu_pitch: IMU pitch angle
            - imu_yaw: IMU yaw angle
            - fl_e_fr_whlspd: Front right wheel speed
            - fl_e_fl_whlspd: Front left wheel speed
            - fl_e_rr_whlspd: Rear right wheel speed
            - fl_e_rl_whlspd: Rear left wheel speed
    """
    if ego_motion_df is None:
        return None
    
    # Check if required columns exist
    required_columns = [
        'timestamp', 'imu_signal_gx', 'imu_signal_gy', 'imu_signal_gz',
        'imu_signal_roll', 'imu_signal_pitch', 'imu_signal_yaw',
        'fl_e_fr_whlspd', 'fl_e_fl_whlspd', 'fl_e_rr_whlspd', 'fl_e_rl_whlspd'
    ]
    
    missing_columns = [col for col in required_columns if col not in ego_motion_df.columns]
    if missing_columns:
        print(f"Warning: Missing columns in ego motion data: {missing_columns}")
        return None
    
    # Extract the required data
    extracted_data = ego_motion_df[required_columns].copy()
    
    # Rename columns for clarity
    extracted_data = extracted_data.rename(columns={
        'imu_signal_gx': 'imu_gx',
        'imu_signal_gy': 'imu_gy', 
        'imu_signal_gz': 'imu_gz',
        'imu_signal_roll': 'imu_roll',
        'imu_signal_pitch': 'imu_pitch',
        'imu_signal_yaw': 'imu_yaw',
        'fl_e_fr_whlspd': 'wheel_vxfr',
        'fl_e_fl_whlspd': 'wheel_vxfl',
        'fl_e_rr_whlspd': 'wheel_vxrr',
        'fl_e_rl_whlspd': 'wheel_vxrl'
    })
    
    print(f"Extracted ego motion data shape: {extracted_data.shape}")
    print(f"Time range: {extracted_data['timestamp'].min()} to {extracted_data['timestamp'].max()}")
    
    return extracted_data

def extract_ublox_data(ublox_df):
    """
    Extract relevant data from ublox DataFrame.
    
    Args:
        ublox_df: DataFrame containing ublox GPS data
        
    Returns:
        pandas.DataFrame: Extracted data with columns:
            - timestamp: System timestamp
            - fix_type: GPS fix type
            - latitude: GPS latitude
            - longitude: GPS longitude
            - height: GPS height
            - vel_north: Velocity north component
            - vel_east: Velocity east component
            - vel_down: Velocity down component
    """
    if ublox_df is None:
        return None
    
    # Check if required columns exist
    required_columns = [
        'timestamp', 'pvat_fix_type', 'pvat_lat', 'pvat_lon', 'pvat_height',
        'pvat_vel_n', 'pvat_vel_e', 'pvat_vel_d'
    ]
    
    missing_columns = [col for col in required_columns if col not in ublox_df.columns]
    if missing_columns:
        print(f"Warning: Missing columns in ublox data: {missing_columns}")
        return None
    
    # Extract the required data
    extracted_data = ublox_df[required_columns].copy()
    
    # Rename columns for clarity
    extracted_data = extracted_data.rename(columns={
        'pvat_fix_type': 'fix_type',
        'pvat_lat': 'latitude',
        'pvat_lon': 'longitude',
        'pvat_height': 'height',
        'pvat_vel_n': 'vel_north',
        'pvat_vel_e': 'vel_east',
        'pvat_vel_d': 'vel_down'
    })

    # conversion
    extracted_data['height'] = extracted_data['height'] * 1e-3  # Convert height from mm to meters
    extracted_data['vel_north'] = extracted_data['vel_north'] * 1e-3  # Convert velocity from mm/s to m/s
    extracted_data['vel_east'] = extracted_data['vel_east'] * 1e-3  # Convert velocity from mm/s to m/s
    extracted_data['vel_down'] = extracted_data['vel_down'] * 1e-3  # Convert velocity from mm/s to m/s

    print(f"Extracted ublox data shape: {extracted_data.shape}")
    print(f"Time range: {extracted_data['timestamp'].min()} to {extracted_data['timestamp'].max()}")
    print(f"Fix types: {sorted(extracted_data['fix_type'].unique())}")
    
    return extracted_data

def extract_vehicle_pose_data(vehicle_pose_df):
    """
    Extract relevant data from vehicle pose DataFrame.
    
    Args:
        vehicle_pose_df: DataFrame containing vehicle pose data
        
    Returns:
        pandas.DataFrame: Extracted data with columns:
            - timestamp: System timestamp
            - position_x: X position
            - position_y: Y position
            - position_z: Z position
            - orientation_w: Quaternion w component
            - orientation_x: Quaternion x component
            - orientation_y: Quaternion y component
            - orientation_z: Quaternion z component
            - roll: Roll angle (computed from quaternion)
            - pitch: Pitch angle (computed from quaternion)
            - yaw: Yaw angle (computed from quaternion)
            - velocity_V_x: Linear velocity X component (vehicle frame)
            - velocity_V_y: Linear velocity Y component (vehicle frame)
            - velocity_V_z: Linear velocity Z component (vehicle frame)
            - rotational_velocity_V_x: Rotational velocity X component
            - rotational_velocity_V_y: Rotational velocity Y component
            - rotational_velocity_V_z: Rotational velocity Z component
    """
    if vehicle_pose_df is None:
        return None
    
    # Import scipy rotation tools
    from scipy.spatial.transform import Rotation as R
    
    # Check if required columns exist
    required_columns = [
        'timestamp', 'position_x', 'position_y', 'position_z',
        'orientation_w', 'orientation_x', 'orientation_y', 'orientation_z',
        'velocity_L_x', 'velocity_L_y', 'velocity_L_z',
        'rotational_velocity_V_x', 'rotational_velocity_V_y', 'rotational_velocity_V_z', 'acceleration_V_x', 'acceleration_V_y', 'acceleration_V_z'
    ]
    
    missing_columns = [col for col in required_columns if col not in vehicle_pose_df.columns]
    if missing_columns:
        print(f"Warning: Missing columns in vehicle pose data: {missing_columns}")
        return None
    
    # Extract the required data
    extracted_data = vehicle_pose_df[required_columns].copy()
    
    # Convert quaternions to Euler angles using scipy
    print("Converting quaternions to Euler angles using scipy.spatial.transform...")
    
    # Create quaternion array in [x, y, z, w] format (scipy convention)
    quaternions = np.column_stack([
        extracted_data['orientation_x'].values,
        extracted_data['orientation_y'].values,
        extracted_data['orientation_z'].values,
        extracted_data['orientation_w'].values
    ])
    
    # Check for invalid quaternions and normalize if needed
    quat_norms = np.linalg.norm(quaternions, axis=1)
    invalid_mask = (quat_norms < 0.9) | (quat_norms > 1.1)
    
    if np.any(invalid_mask):
        print(f"Warning: Found {np.sum(invalid_mask)} invalid quaternions. Normalizing...")
        quaternions[quat_norms > 0] = quaternions[quat_norms > 0] / quat_norms[quat_norms > 0, np.newaxis]
        
        # Set invalid quaternions to identity
        quaternions[quat_norms == 0] = [0, 0, 0, 1]
    
    # Create scipy Rotation object from quaternions
    try:
        rotation_obj = R.from_quat(quaternions)
        
        # Convert to Euler angles (roll, pitch, yaw) using 'xyz' convention (intrinsic rotations)
        # This corresponds to roll-pitch-yaw convention commonly used in robotics
        euler_angles = rotation_obj.as_euler('xyz', degrees=False)
        
        # Add Euler angles to the data
        extracted_data['roll'] = euler_angles[:, 0]
        extracted_data['pitch'] = euler_angles[:, 1] 
        extracted_data['yaw'] = euler_angles[:, 2]
        
        # Get rotation matrices
        rotation_matrices = rotation_obj.as_matrix()
        
        # Convert velocity from local frame (L) to vehicle frame (V)
        # velocity_V = Rot.T @ velocity_L
        velocity_L = np.column_stack([
            extracted_data['velocity_L_x'].values,
            extracted_data['velocity_L_y'].values,
            extracted_data['velocity_L_z'].values
        ])
        
        # Apply rotation transformation: V = R.T @ L
        velocity_V = np.array([
            rotation_matrices[i].T @ velocity_L[i] 
            for i in range(len(velocity_L))
        ])
        
        # Update the velocity columns to vehicle frame
        extracted_data['velocity_V_x'] = velocity_V[:, 0]
        extracted_data['velocity_V_y'] = velocity_V[:, 1]
        extracted_data['velocity_V_z'] = velocity_V[:, 2]
        
        # Remove the original local frame velocity columns
        extracted_data = extracted_data.drop(['velocity_L_x', 'velocity_L_y', 'velocity_L_z'], axis=1)
        
        # Verify rotation matrices are valid
        determinants = np.linalg.det(rotation_matrices)
        invalid_det_mask = np.abs(determinants - 1.0) > 0.1
        
        if np.any(invalid_det_mask):
            print(f"Warning: Found {np.sum(invalid_det_mask)} rotation matrices with invalid determinants")
            
    except Exception as e:
        print(f"Error in quaternion conversion: {str(e)}")
        # Fallback: set angles to zero and keep original velocity
        extracted_data['roll'] = np.zeros(len(extracted_data))
        extracted_data['pitch'] = np.zeros(len(extracted_data))
        extracted_data['yaw'] = np.zeros(len(extracted_data))
        
        # Keep original velocity as vehicle frame (no transformation)
        extracted_data['velocity_V_x'] = extracted_data['velocity_L_x']
        extracted_data['velocity_V_y'] = extracted_data['velocity_L_y']
        extracted_data['velocity_V_z'] = extracted_data['velocity_L_z']
        extracted_data = extracted_data.drop(['velocity_L_x', 'velocity_L_y', 'velocity_L_z'], axis=1)
    
    print(f"Extracted vehicle pose data shape: {extracted_data.shape}")
    print(f"Time range: {extracted_data['timestamp'].min()} to {extracted_data['timestamp'].max()}")
    print(f"Position range - X: {extracted_data['position_x'].min():.3f} to {extracted_data['position_x'].max():.3f} m")
    print(f"Position range - Y: {extracted_data['position_y'].min():.3f} to {extracted_data['position_y'].max():.3f} m")
    print(f"Position range - Z: {extracted_data['position_z'].min():.3f} to {extracted_data['position_z'].max():.3f} m")
    
    # Velocity statistics (vehicle frame)
    velocity_magnitude = np.sqrt(
        extracted_data['velocity_V_x']**2 + 
        extracted_data['velocity_V_y']**2 + 
        extracted_data['velocity_V_z']**2
    )
    print(f"Velocity magnitude range (vehicle frame): {velocity_magnitude.min():.3f} to {velocity_magnitude.max():.3f} m/s")
    
    # Rotational velocity statistics
    rotational_velocity_magnitude = np.sqrt(
        extracted_data['rotational_velocity_V_x']**2 + 
        extracted_data['rotational_velocity_V_y']**2 + 
        extracted_data['rotational_velocity_V_z']**2
    )
    print(f"Rotational velocity magnitude range: {rotational_velocity_magnitude.min():.4f} to {rotational_velocity_magnitude.max():.4f} rad/s")
    
    # Print Euler angle statistics
    print(f"Roll range: {np.degrees(extracted_data['roll'].min()):.1f}° to {np.degrees(extracted_data['roll'].max()):.1f}°")
    print(f"Pitch range: {np.degrees(extracted_data['pitch'].min()):.1f}° to {np.degrees(extracted_data['pitch'].max()):.1f}°")
    print(f"Yaw range: {np.degrees(extracted_data['yaw'].min()):.1f}° to {np.degrees(extracted_data['yaw'].max()):.1f}°")
    
    print(f"Velocity range (vehicle frame) - X: {extracted_data['velocity_V_x'].min():.3f} to {extracted_data['velocity_V_x'].max():.3f} m/s")
    print(f"Velocity range (vehicle frame) - Y: {extracted_data['velocity_V_y'].min():.3f} to {extracted_data['velocity_V_y'].max():.3f} m/s")
    print(f"Velocity range (vehicle frame) - Z: {extracted_data['velocity_V_z'].min():.3f} to {extracted_data['velocity_V_z'].max():.3f} m/s")
    
    return extracted_data

def save_extracted_data(ego_motion_extracted, ublox_extracted, vehicle_pose_extracted, ego_extracted_file, ublox_extracted_file, vehicle_pose_extracted_file):
    """
    Save extracted data to Parquet files.
    
    Args:
        ego_motion_extracted: Extracted ego motion data
        ublox_extracted: Extracted ublox data
        ego_extracted_file: File path to save ego motion data
        ublox_extracted_file: File path to save ublox data
        vehicle_pose_extracted_file: File path to save vehicle pose data
    """
    if ego_extracted_file is None or ublox_extracted_file is None or vehicle_pose_extracted_file is None:
        print("Error: Output file paths must be provided")
        return
    
    if ego_motion_extracted is not None:
        ego_motion_extracted.to_parquet(ego_extracted_file, index=False, engine='pyarrow')
        print(f"Saved ego motion data to: {ego_extracted_file}")

    if ublox_extracted is not None:
        ublox_extracted.to_parquet(ublox_extracted_file, index=False, engine='pyarrow')
        print(f"Saved ublox data to: {ublox_extracted_file}")

    if vehicle_pose_extracted is not None:
        vehicle_pose_extracted.to_parquet(vehicle_pose_extracted_file, index=False, engine='pyarrow')
        print(f"Saved vehicle pose data to: {vehicle_pose_extracted_file}")

def compute_speed_scaling_factor(ego_motion_extracted, ublox_extracted, vehicle_pose_extracted):
    """
    Compute scaling factor between ublox GPS speed and wheel speed.
    
    Args:
        ego_motion_extracted: DataFrame with ego motion data containing:
            - timestamp, wheel_vxfr, wheel_vxfl, wheel_vxrr, wheel_vxrl
        ublox_extracted: DataFrame with ublox data containing:
            - timestamp, vel_north, vel_east, vel_down
            
    Returns:
        dict: Contains scaling factors and statistics
            - scaling_factors: array of scaling factors
            - p50_scaling: median scaling factor
            - mean_scaling: mean scaling factor
            - std_scaling: standard deviation of scaling factor
            - valid_samples: number of valid interpolated samples
    """
    if ego_motion_extracted is None or (ublox_extracted is None and vehicle_pose_extracted is None):
        print("Error: One or both input DataFrames are None")
        return None
    
    # 1. Compute velocity magnitude for wheel speed (average of 2 rear wheels)
    ego_motion_extracted = ego_motion_extracted.copy()
    ego_motion_extracted['wheel_speed_avg'] = (
        ego_motion_extracted['wheel_vxrr'] + 
        ego_motion_extracted['wheel_vxrl']
    ) / 2.0
    
    # Compute wheel speed magnitude (absolute value for scalar speed)
    ego_motion_extracted['wheel_speed_magnitude'] = np.abs(ego_motion_extracted['wheel_speed_avg'])
    
    # 2. Compute velocity magnitude for ublox (3D velocity magnitude)
    if ublox_extracted is None:
        reference_speed_extracted = vehicle_pose_extracted.copy()
        reference_speed_extracted['reference_speed'] = np.sqrt(
            reference_speed_extracted['velocity_V_x']**2 + 
            reference_speed_extracted['velocity_V_y']**2 + 
            reference_speed_extracted['velocity_V_z']**2
        )
    else:
        reference_speed_extracted = ublox_extracted.copy()
        reference_speed_extracted['reference_speed'] = np.sqrt(
            reference_speed_extracted['vel_north']**2 + 
            reference_speed_extracted['vel_east']**2 + 
            reference_speed_extracted['vel_down']**2
        )

    print(f"Ego motion data time range: {ego_motion_extracted['timestamp'].min()} to {ego_motion_extracted['timestamp'].max()}")
    print(f"Ublox data time range: {reference_speed_extracted['timestamp'].min()} to {reference_speed_extracted['timestamp'].max()}")
    print(f"Wheel speed range: {ego_motion_extracted['wheel_speed_magnitude'].min():.3f} to {ego_motion_extracted['wheel_speed_magnitude'].max():.3f}")
    print(f"Ublox speed range: {reference_speed_extracted['reference_speed'].min():.3f} to {reference_speed_extracted['reference_speed'].max():.3f}")

    # 3. For each ublox timestamp, interpolate wheel speed
    ego_timestamps = ego_motion_extracted['timestamp'].values
    ego_wheel_speeds = ego_motion_extracted['wheel_speed_magnitude'].values
    reference_timestamps = reference_speed_extracted['timestamp'].values
    reference_speeds = reference_speed_extracted['reference_speed'].values

    # Find ublox timestamps that fall within ego motion time range
    ego_min_time = ego_timestamps.min()
    ego_max_time = ego_timestamps.max()
    
    valid_reference_mask = (
        (reference_timestamps >= ego_min_time) & 
        (reference_timestamps <= ego_max_time)
    )
    
    valid_reference_timestamps = reference_timestamps[valid_reference_mask]
    valid_reference_speeds = reference_speeds[valid_reference_mask]
    
    print(f"Valid ublox samples within ego motion time range: {len(valid_reference_timestamps)} out of {len(reference_timestamps)}")
    
    if len(valid_reference_timestamps) == 0:
        print("Error: No ublox timestamps overlap with ego motion time range")
        return None
    
    # Interpolate wheel speeds at ublox timestamps
    try:
        # Create interpolation function
        interp_func = interpolate.interp1d(
            ego_timestamps, 
            ego_wheel_speeds, 
            kind='linear', 
            bounds_error=False, 
            fill_value=np.nan
        )
        
        # Interpolate wheel speeds at ublox timestamps
        interpolated_wheel_speeds = interp_func(valid_reference_timestamps)
        
        # Remove any NaN values that might result from interpolation
        valid_mask = ~np.isnan(interpolated_wheel_speeds) & (interpolated_wheel_speeds > 0) & (valid_reference_speeds > 0)
        
        final_wheel_speeds = interpolated_wheel_speeds[valid_mask]
        final_reference_speeds = valid_reference_speeds[valid_mask]
        final_timestamps = valid_reference_timestamps[valid_mask]
        
        print(f"Final valid samples for scaling computation: {len(final_wheel_speeds)}")
        
        if len(final_wheel_speeds) == 0:
            print("Error: No valid samples after interpolation and filtering")
            return None
        
        # 4. Compute scaling factors (reference_speed / wheel_speed)
        scaling_factors = final_reference_speeds / final_wheel_speeds
        
        # Remove outliers (values outside 3 standard deviations)
        mean_scaling = np.mean(scaling_factors)
        std_scaling = np.std(scaling_factors)
        outlier_mask = np.abs(scaling_factors - mean_scaling) <= 3 * std_scaling
        
        scaling_factors_filtered = scaling_factors[outlier_mask]
        
        print(f"Samples after outlier removal: {len(scaling_factors_filtered)} out of {len(scaling_factors)}")
        
        # 5. Compute statistics
        p50_scaling = np.percentile(scaling_factors_filtered, 50)  # Median
        mean_scaling_filtered = np.mean(scaling_factors_filtered)
        std_scaling_filtered = np.std(scaling_factors_filtered)
        p25_scaling = np.percentile(scaling_factors_filtered, 25)
        p75_scaling = np.percentile(scaling_factors_filtered, 75)
        min_scaling = np.min(scaling_factors_filtered)
        max_scaling = np.max(scaling_factors_filtered)
        
        # Create result dictionary
        results = {
            'scaling_factors': scaling_factors_filtered,
            'scaling_factors_all': scaling_factors,  # Before outlier removal
            'p50_scaling': p50_scaling,
            'mean_scaling': mean_scaling_filtered,
            'std_scaling': std_scaling_filtered,
            'p25_scaling': p25_scaling,
            'p75_scaling': p75_scaling,
            'min_scaling': min_scaling,
            'max_scaling': max_scaling,
            'valid_samples': len(scaling_factors_filtered),
            'total_samples': len(scaling_factors),
            'interpolated_wheel_speeds': final_wheel_speeds[outlier_mask],
            'corresponding_reference_speeds': final_reference_speeds[outlier_mask],
            'timestamps': final_timestamps[outlier_mask]
        }
        
        # Print statistics
        print("\nScaling Factor Statistics:")
        print("-" * 40)
        print(f"Valid samples: {results['valid_samples']}")
        print(f"Mean scaling factor: {results['mean_scaling']:.4f}")
        print(f"Median (P50) scaling factor: {results['p50_scaling']:.4f}")
        print(f"Standard deviation: {results['std_scaling']:.4f}")
        print(f"P25 scaling factor: {results['p25_scaling']:.4f}")
        print(f"P75 scaling factor: {results['p75_scaling']:.4f}")
        print(f"Min scaling factor: {results['min_scaling']:.4f}")
        print(f"Max scaling factor: {results['max_scaling']:.4f}")
        
        return results
        
    except Exception as e:
        print(f"Error during interpolation: {str(e)}")
        return None

def plot_scaling_analysis(results, save_path=None):
    """
    Plot scaling factor analysis results.
    
    Args:
        results: Results dictionary from compute_speed_scaling_factor
        save_path: Optional path to save the plot
    """
    import matplotlib.pyplot as plt
    
    if results is None:
        print("No results to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Histogram of scaling factors
    axes[0, 0].hist(results['scaling_factors'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(results['mean_scaling'], color='red', linestyle='--', label=f'Mean: {results["mean_scaling"]:.4f}')
    axes[0, 0].axvline(results['p50_scaling'], color='green', linestyle='--', label=f'Median: {results["p50_scaling"]:.4f}')
    axes[0, 0].set_xlabel('Scaling Factor (Ublox Speed / Wheel Speed)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Scaling Factors')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Scatter plot of wheel speed vs ublox speed
    axes[0, 1].scatter(results['interpolated_wheel_speeds'], results['corresponding_reference_speeds'], 
                       alpha=0.6, s=10)
    # Add ideal line (1:1 ratio with mean scaling factor)
    min_speed = min(results['interpolated_wheel_speeds'].min(), results['corresponding_reference_speeds'].min())
    max_speed = max(results['interpolated_wheel_speeds'].max(), results['corresponding_reference_speeds'].max())
    ideal_line = np.linspace(min_speed, max_speed, 100)
    axes[0, 1].plot(ideal_line, ideal_line * results['mean_scaling'], 'r--', 
                    label=f'Mean scaling: {results["mean_scaling"]:.4f}')
    axes[0, 1].set_xlabel('Interpolated Wheel Speed')
    axes[0, 1].set_ylabel('Ublox Speed')
    axes[0, 1].set_title('Wheel Speed vs Ublox Speed')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Time series of scaling factors
    axes[1, 0].plot(results['timestamps'], results['scaling_factors'], 'b-', alpha=0.7, linewidth=0.5)
    axes[1, 0].axhline(results['mean_scaling'], color='red', linestyle='--', label=f'Mean: {results["mean_scaling"]:.4f}')
    axes[1, 0].axhline(results['p50_scaling'], color='green', linestyle='--', label=f'Median: {results["p50_scaling"]:.4f}')
    axes[1, 0].set_xlabel('Timestamp')
    axes[1, 0].set_ylabel('Scaling Factor')
    axes[1, 0].set_title('Scaling Factor Over Time')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Box plot
    axes[1, 1].boxplot(results['scaling_factors'], vert=True)
    axes[1, 1].set_ylabel('Scaling Factor')
    axes[1, 1].set_title('Scaling Factor Box Plot')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()

def analyze_speed_scaling(ego_motion_extracted, ublox_extracted, vehicle_pose_extracted, output_dir="./analysis", force_extract=False):
    """
    Complete analysis of speed scaling between wheel and GPS data.
    
    Args:
        ego_motion_extracted: Ego motion DataFrame
        ublox_extracted: Ublox GPS DataFrame
        output_dir: Directory to save analysis results
        force_extract: If True, force re-computation; if False, load existing if available
    """
    from pathlib import Path
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Define paths for analysis files
    csv_path = output_path / "scaling_analysis.csv"
    plot_path = output_path / "scaling_analysis_plot.png"
    
    # Check if analysis files exist and force_extract is False
    if not force_extract and csv_path.exists():
        print("Found existing scaling analysis results. Loading from cache...")
        print("=" * 60)
        
        try:
            # Load existing results
            print(f"Loading scaling analysis from: {csv_path}")
            results_df = pd.read_csv(csv_path)
            
            # Reconstruct the results dictionary from the saved CSV
            results = {
                'scaling_factors': results_df['scaling_factor'].values,
                'scaling_factors_all': results_df['scaling_factor'].values,
                'p50_scaling': results_df['scaling_factor'].median(),
                'mean_scaling': results_df['scaling_factor'].mean(),
                'std_scaling': results_df['scaling_factor'].std(),
                'p25_scaling': results_df['scaling_factor'].quantile(0.25),
                'p75_scaling': results_df['scaling_factor'].quantile(0.75),
                'min_scaling': results_df['scaling_factor'].min(),
                'max_scaling': results_df['scaling_factor'].max(),
                'valid_samples': len(results_df),
                'total_samples': len(results_df),
                'interpolated_wheel_speeds': results_df['wheel_speed'].values,
                'corresponding_reference_speeds': results_df['reference_speed'].values,
                'timestamps': results_df['timestamp'].values
            }
            
            print(f"Successfully loaded scaling analysis from cache")
            print(f"Valid samples: {results['valid_samples']}")
            print(f"Median (P50) scaling factor: {results['p50_scaling']:.4f}")
            print(f"Mean scaling factor: {results['mean_scaling']:.4f}")
            
            return results
            
        except Exception as e:
            print(f"Error loading cached scaling analysis: {str(e)}")
            print("Falling back to re-computation...")
            force_extract = True  # Force re-computation if loading fails
    
    elif not force_extract and csv_path.exists():
        print("Warning: Partial scaling analysis found. Re-computing...")
        force_extract = True
    
    if force_extract and csv_path.exists():
        print(f"Force mode: Overwriting existing scaling analysis {csv_path}")
    
    if force_extract and plot_path.exists():
        print(f"Force mode: Overwriting existing plot {plot_path}")
    
    print("Computing speed scaling factors...")
    print("-" * 40)
    
    # Compute scaling factors
    results = compute_speed_scaling_factor(ego_motion_extracted, ublox_extracted, vehicle_pose_extracted)

    if results is None:
        print("Failed to compute scaling factors")
        return None
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'timestamp': results['timestamps'],
        'scaling_factor': results['scaling_factors'],
        'wheel_speed': results['interpolated_wheel_speeds'],
        'reference_speed': results['corresponding_reference_speeds']
    })
    
    results_df.to_csv(csv_path, index=False)
    print(f"Scaling analysis saved to: {csv_path}")
    
    # Plot analysis
    plot_scaling_analysis(results, save_path=plot_path)
    
    return results


def plot_imu_time_series(ego_motion_extracted, vehicle_pose_extracted=None, save_path=None, figsize=(15, 12)):
    """
    Plot IMU time series data with gyroscope and acceleration data.
    
    Args:
        ego_motion_extracted: DataFrame containing IMU data with columns:
            - timestamp: System timestamp
            - imu_gx, imu_gy, imu_gz: IMU acceleration data (m/s²)
            - imu_roll, imu_pitch, imu_yaw: IMU rotation rate data (rad/s)
        vehicle_pose_extracted: Optional DataFrame containing vehicle pose data with columns:
            - timestamp: System timestamp
            - rotational_velocity_V_x, rotational_velocity_V_y, rotational_velocity_V_z: Rotation rates (rad/s)
            - acceleration_V_x, acceleration_V_y, acceleration_V_z: Acceleration (m/s²)
        save_path: Optional path to save the plot
        figsize: Figure size (width, height) in inches
    """
    if ego_motion_extracted is None:
        print("Error: No ego motion data provided")
        return
    
    # Check if required columns exist
    required_columns = ['timestamp', 'imu_gx', 'imu_gy', 'imu_gz', 
                       'imu_roll', 'imu_pitch', 'imu_yaw']
    missing_columns = [col for col in required_columns if col not in ego_motion_extracted.columns]
    
    if missing_columns:
        print(f"Error: Missing columns: {missing_columns}")
        return
    
    # Convert timestamp to datetime for better x-axis formatting
    # Assuming timestamp is in nanoseconds, convert to seconds
    timestamps = ego_motion_extracted['timestamp'].values
    if timestamps.max() > 1e12:  # If timestamp is in nanoseconds
        time_seconds = (timestamps - timestamps.min()) / 1e9  # Convert to seconds from start
    else:  # If timestamp is already in seconds
        time_seconds = timestamps - timestamps.min()
    
    # Process vehicle pose data if provided
    vehicle_pose_time_seconds = None
    vehicle_pose_rot_rates = None
    vehicle_pose_accelerations = None
    
    if vehicle_pose_extracted is not None:
        required_vp_columns = ['timestamp', 'rotational_velocity_V_x', 
                              'rotational_velocity_V_y', 'rotational_velocity_V_z',
                              'acceleration_V_x', 'acceleration_V_y', 'acceleration_V_z']
        missing_vp_columns = [col for col in required_vp_columns if col not in vehicle_pose_extracted.columns]
        
        if not missing_vp_columns:
            vp_timestamps = vehicle_pose_extracted['timestamp'].values
            if vp_timestamps.max() > 1e12:  # If timestamp is in nanoseconds
                vehicle_pose_time_seconds = (vp_timestamps - timestamps.min()) / 1e9  # Align with IMU start time
            else:  # If timestamp is already in seconds
                vehicle_pose_time_seconds = vp_timestamps - timestamps.min()
            
            vehicle_pose_rot_rates = {
                'x': vehicle_pose_extracted['rotational_velocity_V_x'].values,
                'y': vehicle_pose_extracted['rotational_velocity_V_y'].values,
                'z': vehicle_pose_extracted['rotational_velocity_V_z'].values
            }
            
            vehicle_pose_accelerations = {
                'x': vehicle_pose_extracted['acceleration_V_x'].values,
                'y': vehicle_pose_extracted['acceleration_V_y'].values,
                'z': vehicle_pose_extracted['acceleration_V_z'].values
            }
            
            print("Vehicle pose rotation rates and accelerations will be overlaid on IMU plots")
        else:
            print(f"Warning: Vehicle pose data missing columns: {missing_vp_columns}")
    
    # Create figure and subplots (3 rows, 2 columns)
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    fig.suptitle('IMU Time Series Data with Vehicle Pose Comparison', fontsize=16, fontweight='bold')
    
    # Column 1: IMU Acceleration data (gx, gy, gz) with vehicle pose overlay
    # Row 1: gx (acceleration X)
    axes[0, 0].plot(time_seconds, ego_motion_extracted['imu_gx'], 'b-', linewidth=0.8, alpha=0.8, label='IMU gx')
    if vehicle_pose_accelerations is not None:
        axes[0, 0].plot(vehicle_pose_time_seconds, vehicle_pose_accelerations['x'], 'r--', linewidth=1.0, alpha=0.7, label='Vehicle pose ax')
    axes[0, 0].set_title('Acceleration X-axis', fontweight='bold')
    axes[0, 0].set_ylabel('Acceleration (m/s²)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim(time_seconds.min(), time_seconds.max())
    axes[0, 0].legend()
    
    # Row 2: gy (acceleration Y)
    axes[1, 0].plot(time_seconds, ego_motion_extracted['imu_gy'], 'b-', linewidth=0.8, alpha=0.8, label='IMU gy')
    if vehicle_pose_accelerations is not None:
        axes[1, 0].plot(vehicle_pose_time_seconds, vehicle_pose_accelerations['y'], 'r--', linewidth=1.0, alpha=0.7, label='Vehicle pose ay')
    axes[1, 0].set_title('Acceleration Y-axis', fontweight='bold')
    axes[1, 0].set_ylabel('Acceleration (m/s²)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(time_seconds.min(), time_seconds.max())
    axes[1, 0].legend()
    
    # Row 3: gz (acceleration Z)
    axes[2, 0].plot(time_seconds, ego_motion_extracted['imu_gz']-9.80665, 'b-', linewidth=0.8, alpha=0.8, label='IMU gz')
    if vehicle_pose_accelerations is not None:
        axes[2, 0].plot(vehicle_pose_time_seconds, vehicle_pose_accelerations['z'], 'r--', linewidth=1.0, alpha=0.7, label='Vehicle pose az')
    axes[2, 0].set_title('Acceleration Z-axis', fontweight='bold')
    axes[2, 0].set_ylabel('Acceleration (m/s²)')
    axes[2, 0].set_xlabel('Time (seconds)')
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].set_xlim(time_seconds.min(), time_seconds.max())
    axes[2, 0].legend()
    
    # Column 2: IMU Rotation rate data (roll, pitch, yaw) with vehicle pose overlay
    # Row 1: roll rate
    axes[0, 1].plot(time_seconds, ego_motion_extracted['imu_roll'], 'b-', linewidth=0.8, alpha=0.8, label='IMU roll rate')
    if vehicle_pose_rot_rates is not None:
        axes[0, 1].plot(vehicle_pose_time_seconds, vehicle_pose_rot_rates['x'], 'r--', linewidth=1.0, alpha=0.7, label='Vehicle pose ωx')
    axes[0, 1].set_title('Angular Velocity Roll (X-axis)', fontweight='bold')
    axes[0, 1].set_ylabel('Rate (rad/s)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(time_seconds.min(), time_seconds.max())
    axes[0, 1].legend()
    
    # Row 2: pitch rate
    axes[1, 1].plot(time_seconds, ego_motion_extracted['imu_pitch'], 'b-', linewidth=0.8, alpha=0.8, label='IMU pitch rate')
    if vehicle_pose_rot_rates is not None:
        axes[1, 1].plot(vehicle_pose_time_seconds, vehicle_pose_rot_rates['y'], 'r--', linewidth=1.0, alpha=0.7, label='Vehicle pose ωy')
    axes[1, 1].set_title('Angular Velocity Pitch (Y-axis)', fontweight='bold')
    axes[1, 1].set_ylabel('Rate (rad/s)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(time_seconds.min(), time_seconds.max())
    axes[1, 1].legend()
    
    # Row 3: yaw rate
    axes[2, 1].plot(time_seconds, ego_motion_extracted['imu_yaw'], 'b-', linewidth=0.8, alpha=0.8, label='IMU yaw rate')
    if vehicle_pose_rot_rates is not None:
        axes[2, 1].plot(vehicle_pose_time_seconds, vehicle_pose_rot_rates['z'], 'r--', linewidth=1.0, alpha=0.7, label='Vehicle pose ωz')
    axes[2, 1].set_title('Angular Velocity Yaw (Z-axis)', fontweight='bold')
    axes[2, 1].set_ylabel('Rate (rad/s)')
    axes[2, 1].set_xlabel('Time (seconds)')
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].set_xlim(time_seconds.min(), time_seconds.max())
    axes[2, 1].legend()
    
    # Add statistics text to each subplot
    imu_data_info = [
        ('imu_gx', 'IMU', 'x'),
        ('imu_gy', 'IMU', 'y'), 
        ('imu_gz', 'IMU', 'z'),
        ('imu_roll', 'IMU', 'x'),
        ('imu_pitch', 'IMU', 'y'),
        ('imu_yaw', 'IMU', 'z')
    ]
    
    for i, (col_name, sensor, axis) in enumerate(imu_data_info):
        row = i % 3
        col = i // 3
        
        data = ego_motion_extracted[col_name]
        stats_text = f'IMU μ={data.mean():.3f}, σ={data.std():.3f}\nmin={data.min():.3f}, max={data.max():.3f}'
        
        # Add vehicle pose statistics if available
        if col == 0 and vehicle_pose_accelerations is not None:  # Left column (accelerations)
            vp_data = vehicle_pose_accelerations[axis]
            stats_text += f'\nVP μ={vp_data.mean():.3f}, σ={vp_data.std():.3f}'
        elif col == 1 and vehicle_pose_rot_rates is not None:  # Right column (rotation rates)
            vp_data = vehicle_pose_rot_rates[axis]
            stats_text += f'\nVP μ={vp_data.mean():.3f}, σ={vp_data.std():.3f}'
        
        # Position text box in upper right corner
        axes[row, col].text(0.98, 0.98, stats_text, 
                           transform=axes[row, col].transAxes,
                           fontsize=8, verticalalignment='top', horizontalalignment='right',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the plot if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"IMU time series plot saved to: {save_path}")
    
    # Display the plot
    plt.show()
    
    # Print summary statistics
    print("\nIMU Data Summary Statistics:")
    print("="*60)
    print(f"{'Parameter':<12} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-"*60)
    
    for param in ['imu_gx', 'imu_gy', 'imu_gz', 'imu_roll', 'imu_pitch', 'imu_yaw']:
        data = ego_motion_extracted[param]
        print(f"{param:<12} {data.mean():>9.4f} {data.std():>9.4f} {data.min():>9.4f} {data.max():>9.4f}")
    
    # Print vehicle pose statistics if available
    if vehicle_pose_rot_rates is not None and vehicle_pose_accelerations is not None:
        print("\nVehicle Pose Statistics:")
        print("="*60)
        print(f"{'Parameter':<12} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
        print("-"*60)
        
        # Rotation rates
        for i, component in enumerate(['x', 'y', 'z']):
            data = vehicle_pose_rot_rates[component]
            param_name = f"rot_vel_{component}"
            print(f"{param_name:<12} {data.mean():>9.4f} {data.std():>9.4f} {data.min():>9.4f} {data.max():>9.4f}")
        
        # Accelerations
        for i, component in enumerate(['x', 'y', 'z']):
            data = vehicle_pose_accelerations[component]
            param_name = f"accel_{component}"
            print(f"{param_name:<12} {data.mean():>9.4f} {data.std():>9.4f} {data.min():>9.4f} {data.max():>9.4f}")
        
        # Compute correlations between IMU and vehicle pose data
        print("\nCorrelations between IMU and Vehicle Pose:")
        print("-"*60)
        
        from scipy import interpolate
        
        try:
            # Correlations for rotation rates
            imu_rot_components = ['imu_roll', 'imu_pitch', 'imu_yaw']
            vp_rot_components = ['x', 'y', 'z']
            
            print("Rotation Rate Correlations:")
            for i, (imu_comp, vp_comp) in enumerate(zip(imu_rot_components, vp_rot_components)):
                # Interpolate vehicle pose data to IMU timestamps
                interp_func = interpolate.interp1d(
                    vehicle_pose_time_seconds, 
                    vehicle_pose_rot_rates[vp_comp], 
                    kind='linear', 
                    bounds_error=False, 
                    fill_value=np.nan
                )
                
                vp_interp = interp_func(time_seconds)
                
                # Remove NaN values
                valid_mask = ~np.isnan(vp_interp)
                
                if np.sum(valid_mask) > 10:  # Need at least 10 points for correlation
                    correlation = np.corrcoef(
                        ego_motion_extracted[imu_comp].values[valid_mask], 
                        vp_interp[valid_mask]
                    )[0, 1]
                    
                    print(f"  {imu_comp} vs VP ω{vp_comp}: {correlation:.4f}")
                else:
                    print(f"  {imu_comp} vs VP ω{vp_comp}: insufficient overlap")
            
            # Correlations for accelerations
            imu_acc_components = ['imu_gx', 'imu_gy', 'imu_gz']
            vp_acc_components = ['x', 'y', 'z']
            
            print("Acceleration Correlations:")
            for i, (imu_comp, vp_comp) in enumerate(zip(imu_acc_components, vp_acc_components)):
                # Interpolate vehicle pose data to IMU timestamps
                interp_func = interpolate.interp1d(
                    vehicle_pose_time_seconds, 
                    vehicle_pose_accelerations[vp_comp], 
                    kind='linear', 
                    bounds_error=False, 
                    fill_value=np.nan
                )
                
                vp_interp = interp_func(time_seconds)
                
                # Remove NaN values
                valid_mask = ~np.isnan(vp_interp)
                
                if np.sum(valid_mask) > 10:  # Need at least 10 points for correlation
                    correlation = np.corrcoef(
                        ego_motion_extracted[imu_comp].values[valid_mask], 
                        vp_interp[valid_mask]
                    )[0, 1]
                    
                    print(f"  {imu_comp} vs VP a{vp_comp}: {correlation:.4f}")
                else:
                    print(f"  {imu_comp} vs VP a{vp_comp}: insufficient overlap")
                    
        except Exception as e:
            print(f"Error computing correlations: {str(e)}")

def plot_wheel_speed_time_series(ego_motion_extracted, save_path=None, figsize=(15, 8)):
    """
    Plot wheel speed time series data.
    
    Args:
        ego_motion_extracted: DataFrame containing wheel speed data
        save_path: Optional path to save the plot
        figsize: Figure size (width, height) in inches
    """
    if ego_motion_extracted is None:
        print("Error: No ego motion data provided")
        return
    
    # Check if required columns exist
    wheel_columns = ['wheel_vxfr', 'wheel_vxfl', 'wheel_vxrr', 'wheel_vxrl']
    missing_columns = [col for col in wheel_columns if col not in ego_motion_extracted.columns]
    
    if missing_columns:
        print(f"Error: Missing wheel speed columns: {missing_columns}")
        return
    
    # Convert timestamp
    timestamps = ego_motion_extracted['timestamp'].values
    if timestamps.max() > 1e12:
        time_seconds = (timestamps - timestamps.min()) / 1e9
    else:
        time_seconds = timestamps - timestamps.min()
    
    # Create figure with subplots (2 rows, 2 columns)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Wheel Speed Time Series Data', fontsize=16, fontweight='bold')
    
    wheel_labels = ['Front Right (FR)', 'Front Left (FL)', 'Rear Right (RR)', 'Rear Left (RL)']
    colors = ['blue', 'green', 'red', 'orange']
    
    for i, (col, label, color) in enumerate(zip(wheel_columns, wheel_labels, colors)):
        row = i // 2
        col_idx = i % 2
        
        axes[row, col_idx].plot(time_seconds, ego_motion_extracted[col], color=color, linewidth=0.8, alpha=0.8)
        axes[row, col_idx].set_title(f'{label} Wheel Speed', fontweight='bold')
        axes[row, col_idx].set_ylabel('Speed (m/s)')
        axes[row, col_idx].grid(True, alpha=0.3)
        axes[row, col_idx].set_xlim(time_seconds.min(), time_seconds.max())
        
        if row == 1:  # Bottom row
            axes[row, col_idx].set_xlabel('Time (seconds)')
        
        # Add statistics
        data = ego_motion_extracted[col]
        stats_text = f'μ={data.mean():.2f}, σ={data.std():.2f}\nmin={data.min():.2f}, max={data.max():.2f}'
        axes[row, col_idx].text(0.98, 0.98, stats_text, 
                               transform=axes[row, col_idx].transAxes,
                               fontsize=8, verticalalignment='top', horizontalalignment='right',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Wheel speed time series plot saved to: {save_path}")
    
    plt.show()




import numpy as np
from utils_numpy_filter import NUMPYIEKF
from pathlib import Path
import matplotlib.pyplot as plt

def _run_iekf_with_inputs(t, u, v_mes, initial_position, initial_velocity, initial_orientation,
                          measurement_covariances, filter_params, save_results=True, output_dir="./iekf_results", verbose=False):
    # Set initial conditions
    if initial_position is None:
        initial_position = np.array([0.0, 0.0, 0.0])
    
    if initial_orientation is None:
        # Use first IMU attitude measurement
        initial_orientation = np.array([0.0, 0.0, 0.0])

    if initial_velocity is None:
        initial_velocity = v_mes[0]
    
    print(f"Initial position: {initial_position}")
    print(f"Initial velocity: {initial_velocity} m/s")
    print(f"Initial orientation (RPY): {np.degrees(initial_orientation)} degrees")

    # Initialize IEKF with custom parameters if provided
    if filter_params is not None:
        iekf = NUMPYIEKF(filter_params, verbose=verbose)
    else:
        # Use default parameters with some adjustments for vehicle data
        class VehicleParameters(NUMPYIEKF.Parameters):
            cov_omega = 1.72e-4
            cov_acc = 1.06e-3
            cov_b_omega = 9.47e-09
            cov_b_acc = 9.46e-07
            cov_Rot_c_i = 1.22e-08
            cov_t_c_i = 9.23e-09
 

            cov_Rot0 = 9.52e-07
            cov_v0 = 9.79e-02
            cov_b_omega0 = 8.55e-09
            cov_b_acc0 = 1.07e-03
            cov_Rot_c_i0 = 8.54e-06
            cov_t_c_i0 = 9.68e-03
            
            cov_long = 0.1
            cov_lat = 1
            cov_up = 10
            
            
        iekf = NUMPYIEKF(VehicleParameters, verbose=verbose)

    # Prepare measurement covariances for zero lateral/vertical velocity constraints
    N = len(t)
    if measurement_covariances is None:
        measurement_covariances = np.tile([iekf.cov_long,iekf.cov_lat, iekf.cov_up], (N, 1))

    print("Running IEKF...")


    # Run the filter
    try:
        Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i = iekf.run(
            t=t,
            u=u, 
            measurements_covs=measurement_covariances,
            v_mes=v_mes,
            p_mes0=initial_position,
            N=N,
            ang0=initial_orientation,
        )

        # Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i = iekf.run_with_bias_reset(
        #     t=t,
        #     u=u, 
        #     measurements_covs=measurement_covariances,
        #     v_mes=v_mes,
        #     p_mes0=initial_position,
        #     N=N,
        #     ang0=initial_orientation
        # )
        
        print("IEKF completed successfully!")
        
        # Convert rotation matrices to Euler angles for easier interpretation
        euler_angles = np.zeros((N, 3))
        v_ego = np.zeros((N, 3))
        omega = np.zeros((N, 3))
        accel = np.zeros((N, 3))
        rot_euler = np.zeros((N, 3))
        for i in range(N):
            euler_angles[i] = NUMPYIEKF.to_rpy(Rot[i].dot(Rot_c_i[i]))
            omega_imu = u[i, 0:3] - b_omega[i]
            omega[i] = Rot_c_i[i].T.dot(omega_imu)
            v_ego[i] = Rot_c_i[i].T.dot(Rot[i].T @ v[i] + NUMPYIEKF.skew(omega_imu).dot(t_c_i[i]))
            accel[i] = Rot_c_i[i].T.dot(u[i, 3:6] - b_acc[i]) + Rot[i].T @ np.array([0, 0, -9.80665])
            rot_euler[i] = NUMPYIEKF.to_rpy(Rot_c_i[i])

        # Prepare results dictionary
        results = {
            'timestamps': t,
            'Rot': Rot,
            'v': v_ego,
            'p': p, 
            'b_omega': b_omega,
            'b_acc': b_acc,
            'Rot_c_i': rot_euler,
            't_c_i': t_c_i,
            'euler_angles': euler_angles,
            'wheel_speed_avg': v_mes[:, 0],
            'measurement_covariances': measurement_covariances,
            'omega': omega,
            'accel': accel
        }

        # Print summary statistics
        print("\nIEKF Results Summary:")
        print("-" * 40)
        print(f"Total distance traveled: {np.linalg.norm(p[-1] - p[0]):.2f} m")
        print(f"Final position: [{p[-1, 0]:.2f}, {p[-1, 1]:.2f}, {p[-1, 2]:.2f}] m")
        print(f"Final velocity: [{v[-1, 0]:.2f}, {v[-1, 1]:.2f}, {v[-1, 2]:.2f}] m/s")
        print(f"Final orientation: [{np.degrees(euler_angles[-1, 0]):.1f}, {np.degrees(euler_angles[-1, 1]):.1f}, {np.degrees(euler_angles[-1, 2]):.1f}] degrees")
        print(f"Gyro bias: [{b_omega[-1, 0]:.4f}, {b_omega[-1, 1]:.4f}, {b_omega[-1, 2]:.4f}] rad/s")
        print(f"Acc bias: [{b_acc[-1, 0]:.2f}, {b_acc[-1, 1]:.2f}, {b_acc[-1, 2]:.2f}] m/s²")

        # print time and u[:,4] statistics
        for i in range(0, 800, 40):
            print(f"Time: {t[i]:.2f} s, {u[i, 4]:.2f}, {accel[i, 2]:.2f}")

        # Save results if requested
        if save_results:
            save_iekf_results(results, output_dir)
        
        return results
        
    except Exception as e:
        print(f"Error running IEKF: {str(e)}")
        return None

def run_iekf_with_vehicle_pose(vehicle_pose_extracted, initial_position=None, initial_velocity=None, 
                              initial_orientation=None, measurement_covariances=None, 
                              filter_params=None, save_results=True, output_dir="./iekf_results", verbose=False):

    if vehicle_pose_extracted is None or vehicle_pose_extracted.empty:
        print("Error: No ego motion data provided")
        return None
    
    print("Preparing IEKF data from ego motion measurements...")
    
    # Convert timestamps to seconds
    timestamps = vehicle_pose_extracted['timestamp'].values
    if timestamps.max() > 1e12:  # If in nanoseconds
        t = timestamps / 1e9  # Convert to seconds from start
    else:
        t = timestamps
    
    N = len(t)
    print(f"Processing {N} data points over {(t[-1] - t[0]):.2f} seconds")
    
    # Prepare IMU data [acc_x, acc_y, acc_z,gyro_x, gyro_y, gyro_z,]
    # Note: We don't have direct accelerometer measurements, so we'll use finite differences of velocity
    # or set to zero and let the filter estimate from constraints
    u = np.zeros((N, 6))
    u[:, 0] = vehicle_pose_extracted['rotational_velocity_V_x'].values  # rotation rate x
    u[:, 1] = vehicle_pose_extracted['rotational_velocity_V_y'].values  # rotation rate y
    u[:, 2] = vehicle_pose_extracted['rotational_velocity_V_z'].values  # rotation rate z
    u[:, 3] = vehicle_pose_extracted['acceleration_V_x'].values  # accel X
    u[:, 4] = vehicle_pose_extracted['acceleration_V_y'].values  # accel Y
    u[:, 5] = vehicle_pose_extracted['acceleration_V_z'].values + 9.80665  # accel Z

    velocity = vehicle_pose_extracted[['velocity_V_x', 'velocity_V_y', 'velocity_V_z']].values
    velocity_magnitude = np.linalg.norm(velocity, axis=1)

    v_mes = np.column_stack((velocity_magnitude, np.zeros((velocity_magnitude.size, 2))))

    return _run_iekf_with_inputs(t, u, v_mes, initial_position, initial_velocity, initial_orientation, measurement_covariances, filter_params, save_results, output_dir, verbose)


def run_iekf_with_ego_motion(ego_motion_extracted, wheel_speed_scaling = 1.0, initial_position=None, initial_velocity=None, 
                             initial_orientation=None, measurement_covariances=None, 
                             filter_params=None, save_results=True, output_dir="./iekf_results", verbose=False):
    """
    Run IEKF with extracted ego motion data.

    Args:
        ego_motion_extracted: DataFrame with columns:
            - timestamp: System timestamp
            - imu_gx, imu_gy, imu_gz: Gyroscope data (rad/s)
            - imu_roll, imu_pitch, imu_yaw: Attitude data (rad)
            - wheel_vxfr, wheel_vxfl, wheel_vxrr, wheel_vxrl: Wheel speeds (m/s)
        initial_position: Initial position [x, y, z] in meters (default: [0, 0, 0])
        initial_velocity: Initial velocity [vx, vy, vz] in m/s (default: [0, 0, 0])
        initial_orientation: Initial orientation [roll, pitch, yaw] in radians 
                           (default: uses first IMU measurement)
        measurement_covariances: Measurement covariances for zero velocity constraints
                               (default: adaptive based on wheel speed)
        filter_params: Custom filter parameters (default: uses NUMPYIEKF.Parameters())
        save_results: Whether to save results to CSV files
        output_dir: Directory to save results
        
    Returns:
        dict: IEKF results containing:
            - timestamps: Time vector
            - Rot: Rotation matrices (N, 3, 3)
            - v: Velocities (N, 3)
            - p: Positions (N, 3)
            - b_omega: Gyro biases (N, 3)
            - b_acc: Accelerometer biases (N, 3)
            - Rot_c_i: Car-to-IMU rotations (N, 3, 3)
            - t_c_i: Car-to-IMU translations (N, 3)
            - euler_angles: Roll, pitch, yaw (N, 3)
    """
    if ego_motion_extracted is None or ego_motion_extracted.empty:
        print("Error: No ego motion data provided")
        return None
    
    print("Preparing IEKF data from ego motion measurements...")
    
    # Convert timestamps to seconds
    timestamps = ego_motion_extracted['timestamp'].values
    if timestamps.max() > 1e12:  # If in nanoseconds
        t = timestamps / 1e9  # Convert to seconds from start
    else:
        t = timestamps
    
    N = len(t)
    print(f"Processing {N} data points over {t[-1]:.2f} seconds")
    
    # Prepare IMU data [acc_x, acc_y, acc_z,gyro_x, gyro_y, gyro_z,]
    # Note: We don't have direct accelerometer measurements, so we'll use finite differences of velocity
    # or set to zero and let the filter estimate from constraints
    u = np.zeros((N, 6))
    u[:, 0] = ego_motion_extracted['imu_roll'].values  # rotation rate x
    u[:, 1] = ego_motion_extracted['imu_pitch'].values  # rotation rate y
    u[:, 2] = ego_motion_extracted['imu_yaw'].values  # rotation rate z
    u[:, 3] = ego_motion_extracted['imu_gx'].values  # accel X
    u[:, 4] = ego_motion_extracted['imu_gy'].values  # accel Y
    u[:, 5] = ego_motion_extracted['imu_gz'].values  # accel Z


    # For accelerometer, we can estimate from wheel speed changes or set to nominal values
    # Here we'll use a simple approach - differentiate wheel speed for longitudinal acceleration
    wheel_speed_avg = (ego_motion_extracted['wheel_vxrr'] + ego_motion_extracted['wheel_vxrl']).values / 2.0 * wheel_speed_scaling
    v_mes = np.column_stack((wheel_speed_avg, np.zeros((wheel_speed_avg.size, 2))))

    return _run_iekf_with_inputs(t, u, v_mes, initial_position, initial_velocity, initial_orientation, measurement_covariances, filter_params, save_results, output_dir, verbose)


def save_iekf_results(results, output_dir):
    """
    Save IEKF results to CSV files.
    
    Args:
        results: IEKF results dictionary
        output_dir: Directory to save files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save trajectory (position, velocity, orientation)
    trajectory_data = {
        'timestamp': results['timestamps'],
        'pos_x': results['p'][:, 0],
        'pos_y': results['p'][:, 1], 
        'pos_z': results['p'][:, 2],
        'vel_x': results['v'][:, 0],
        'vel_y': results['v'][:, 1],
        'vel_z': results['v'][:, 2],
        'roll': results['euler_angles'][:, 0],
        'pitch': results['euler_angles'][:, 1],
        'yaw': results['euler_angles'][:, 2],
        'wheel_speed': results['wheel_speed_avg']
    }
    
    import pandas as pd
    trajectory_df = pd.DataFrame(trajectory_data)
    trajectory_file = output_path / "iekf_trajectory.csv"
    trajectory_df.to_csv(trajectory_file, index=False)
    print(f"Trajectory saved to: {trajectory_file}")
    
    # Save bias estimates
    bias_data = {
        'timestamp': results['timestamps'],
        'gyro_bias_x': results['b_omega'][:, 0],
        'gyro_bias_y': results['b_omega'][:, 1],
        'gyro_bias_z': results['b_omega'][:, 2],
        'acc_bias_x': results['b_acc'][:, 0], 
        'acc_bias_y': results['b_acc'][:, 1],
        'acc_bias_z': results['b_acc'][:, 2]
    }
    
    bias_df = pd.DataFrame(bias_data)
    bias_file = output_path / "iekf_biases.csv"
    bias_df.to_csv(bias_file, index=False)
    print(f"Bias estimates saved to: {bias_file}")

def plot_iekf_results(results, save_path=None, figsize=(20, 15)):
    """
    Plot IEKF estimation results.
    
    Args:
        results: IEKF results dictionary
        save_path: Optional path to save the plot
        figsize: Figure size (width, height) in inches
    """
    if results is None:
        print("No results to plot")
        return
    
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    fig.suptitle('IEKF Estimation Results', fontsize=16, fontweight='bold')
    
    t = results['timestamps'] - results['timestamps'][0]
    
    # Position plot
    axes[0, 0].plot(t, results['p'][:, 0], 'b-', label='X', linewidth=1.5)
    axes[0, 0].plot(t, results['p'][:, 1], 'g-', label='Y', linewidth=1.5)
    axes[0, 0].plot(t, results['p'][:, 2], 'r-', label='Z', linewidth=1.5)
    axes[0, 0].set_title('Position', fontweight='bold')
    axes[0, 0].set_ylabel('Position (m)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Velocity plot
    axes[0, 1].plot(t, results['v'][:, 0], 'b-', label='Vx', linewidth=1.5)
    axes[0, 1].plot(t, results['v'][:, 1], 'g-', label='Vy', linewidth=1.5)
    axes[0, 1].plot(t, results['v'][:, 2], 'r-', label='Vz', linewidth=1.5)
    axes[0, 1].plot(t, results['wheel_speed_avg'], 'k--', label='Wheel Speed', linewidth=1.0, alpha=0.7)
    axes[0, 1].set_title('Velocity', fontweight='bold')
    axes[0, 1].set_ylabel('Velocity (m/s)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Orientation plot (in degrees)
    euler_deg = np.degrees(results['euler_angles'])
    axes[0, 2].plot(t, euler_deg[:, 0], 'b-', label='Roll', linewidth=1.5)
    axes[0, 2].plot(t, euler_deg[:, 1], 'g-', label='Pitch', linewidth=1.5) 
    axes[0, 2].plot(t, euler_deg[:, 2], 'r-', label='Yaw', linewidth=1.5)
    axes[0, 2].set_title('Orientation', fontweight='bold')
    axes[0, 2].set_ylabel('Angle (degrees)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Gyro bias plot
    axes[1, 0].plot(t, results['b_omega'][:, 0], 'b-', label='Bias X', linewidth=1.5)
    axes[1, 0].plot(t, results['b_omega'][:, 1], 'g-', label='Bias Y', linewidth=1.5)
    axes[1, 0].plot(t, results['b_omega'][:, 2], 'r-', label='Bias Z', linewidth=1.5)
    axes[1, 0].set_title('Gyroscope Bias', fontweight='bold')
    axes[1, 0].set_ylabel('Bias (rad/s)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Accelerometer bias plot
    axes[1, 1].plot(t, results['b_acc'][:, 0], 'b-', label='Bias X', linewidth=1.5)
    axes[1, 1].plot(t, results['b_acc'][:, 1], 'g-', label='Bias Y', linewidth=1.5)
    axes[1, 1].plot(t, results['b_acc'][:, 2], 'r-', label='Bias Z', linewidth=1.5)
    axes[1, 1].set_title('Accelerometer Bias', fontweight='bold')
    axes[1, 1].set_ylabel('Bias (m/s²)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Car-to-IMU rotation (Rot_c_i) in degrees
    rot_c_i_deg = np.degrees(results['Rot_c_i'])
    axes[1, 2].plot(t, rot_c_i_deg[:, 0], 'b-', label='Roll', linewidth=1.5)
    axes[1, 2].plot(t, rot_c_i_deg[:, 1], 'g-', label='Pitch', linewidth=1.5)
    axes[1, 2].plot(t, rot_c_i_deg[:, 2], 'r-', label='Yaw', linewidth=1.5)
    axes[1, 2].set_title('Car-to-IMU Rotation', fontweight='bold')
    axes[1, 2].set_ylabel('Angle (degrees)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    # 2D trajectory plot
    axes[2, 0].plot(results['p'][:, 0], results['p'][:, 1], 'b-', linewidth=2)
    axes[2, 0].plot(results['p'][0, 0], results['p'][0, 1], 'go', markersize=8, label='Start')
    axes[2, 0].plot(results['p'][-1, 0], results['p'][-1, 1], 'ro', markersize=8, label='End')
    axes[2, 0].set_title('2D Trajectory', fontweight='bold')
    axes[2, 0].set_xlabel('X (m)')
    axes[2, 0].set_ylabel('Y (m)')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].axis('equal')
    
    # Angular velocity (omega) - bias corrected
    if 'omega' in results:
        axes[2, 1].plot(t, results['omega'][:, 0], 'b-', label='ωx', linewidth=1.5)
        axes[2, 1].plot(t, results['omega'][:, 1], 'g-', label='ωy', linewidth=1.5)
        axes[2, 1].plot(t, results['omega'][:, 2], 'r-', label='ωz', linewidth=1.5)
        axes[2, 1].set_title('Corrected Angular Velocity', fontweight='bold')
        axes[2, 1].set_xlabel('Time (s)')
        axes[2, 1].set_ylabel('Angular Velocity (rad/s)')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
    else:
        axes[2, 1].text(0.5, 0.5, 'Angular velocity\ndata not available', 
                       transform=axes[2, 1].transAxes, ha='center', va='center')
        axes[2, 1].set_title('Corrected Angular Velocity', fontweight='bold')
    
    # Acceleration (accel) - bias corrected
    if 'accel' in results:
        axes[2, 2].plot(t, results['accel'][:, 0], 'b-', label='ax', linewidth=1.5)
        axes[2, 2].plot(t, results['accel'][:, 1], 'g-', label='ay', linewidth=1.5)
        axes[2, 2].plot(t, results['accel'][:, 2], 'r-', label='az', linewidth=1.5)
        axes[2, 2].set_title('Corrected Acceleration', fontweight='bold')
        axes[2, 2].set_xlabel('Time (s)')
        axes[2, 2].set_ylabel('Acceleration (m/s²)')
        axes[2, 2].legend()
        axes[2, 2].grid(True, alpha=0.3)
    else:
        axes[2, 2].text(0.5, 0.5, 'Acceleration\ndata not available', 
                       transform=axes[2, 2].transAxes, ha='center', va='center')
        axes[2, 2].set_title('Corrected Acceleration', fontweight='bold')
    
    # Add statistics text boxes to key plots
    statistics_plots = [
        (0, 0, 'p', 'Position'),
        (0, 1, 'v', 'Velocity'),
        (1, 0, 'b_omega', 'Gyro Bias'),
        (1, 1, 'b_acc', 'Acc Bias')
    ]
    
    for row, col, key, name in statistics_plots:
        if key in results:
            data = results[key]
            if len(data.shape) > 1:
                # Multi-dimensional data
                stats_text = f'{name} RMS:\nX: {np.sqrt(np.mean(data[:, 0]**2)):.3f}\nY: {np.sqrt(np.mean(data[:, 1]**2)):.3f}\nZ: {np.sqrt(np.mean(data[:, 2]**2)):.3f}'
            else:
                # 1D data
                stats_text = f'{name} RMS: {np.sqrt(np.mean(data**2)):.3f}'
            
            axes[row, col].text(0.02, 0.98, stats_text, 
                               transform=axes[row, col].transAxes,
                               fontsize=8, verticalalignment='top', horizontalalignment='left',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"IEKF results plot saved to: {save_path}")
    
    plt.show()
    
    # Print additional summary statistics
    if 'omega' in results and 'accel' in results:
        print("\nAdditional IEKF Statistics:")
        print("-" * 50)
        
        omega_rms = np.sqrt(np.mean(results['omega']**2, axis=0))
        accel_rms = np.sqrt(np.mean(results['accel']**2, axis=0))
        rot_c_i_std = np.std(results['Rot_c_i'], axis=0)
        
        print(f"Corrected angular velocity RMS: [{omega_rms[0]:.4f}, {omega_rms[1]:.4f}, {omega_rms[2]:.4f}] rad/s")
        print(f"Corrected acceleration RMS: [{accel_rms[0]:.2f}, {accel_rms[1]:.2f}, {accel_rms[2]:.2f}] m/s²")
        print(f"Car-to-IMU rotation std: [{np.degrees(rot_c_i_std[0]):.2f}, {np.degrees(rot_c_i_std[1]):.2f}, {np.degrees(rot_c_i_std[2]):.2f}] degrees")
        
        # Check if the corrected measurements look reasonable
        if np.any(omega_rms > 10):  # Very high angular rates
            print("WARNING: Unusually high angular velocities detected!")
        
        if np.any(accel_rms > 50):  # Very high accelerations
            print("WARNING: Unusually high accelerations detected!")
        
        if np.any(rot_c_i_std > np.radians(10)):  # Car-to-IMU rotation changing too much
            print("WARNING: Car-to-IMU rotation estimates are unstable!")

def plot_iekf_vs_vehicle_pose(iekf_results, vehicle_pose_extracted, save_path=None, figsize=(20, 16)):
    """
    Plot IEKF results compared against vehicle pose data with proper temporal and spatial alignment.
    
    Args:
        iekf_results: IEKF results dictionary
        vehicle_pose_extracted: Vehicle pose DataFrame
        ego_motion_extracted: Ego motion DataFrame (for timestamp alignment)
        save_path: Optional path to save the plot
        figsize: Figure size (width, height) in inches
    """
    if iekf_results is None or vehicle_pose_extracted is None:
        print("Error: IEKF results or vehicle pose data not provided")
        return
    
    from scipy import interpolate
    from scipy.spatial.transform import Rotation as R
    from scipy.spatial.transform import Slerp
    
    print("Preparing IEKF vs Vehicle Pose comparison with SLERP interpolation...")
    
    # Get IEKF timestamps (relative to start)
    iekf_time = iekf_results['timestamps']

    
    # Get vehicle pose timestamps and convert to same reference as IEKF
    vp_timestamps = vehicle_pose_extracted['timestamp'].values

    if vp_timestamps.max() > 1e12:  # If in nanoseconds
        vp_time_seconds = vp_timestamps / 1e9  # Convert to seconds
    else:
        vp_time_seconds = vp_timestamps  # Already in seconds
    
    # Find the temporal overlap between IEKF and vehicle pose data
    iekf_start_time = iekf_time[0]
    iekf_end_time = iekf_time[-1]
    vp_start_time = vp_time_seconds[0]
    vp_end_time = vp_time_seconds[-1]
    
    overlap_start = max(iekf_start_time, vp_start_time)
    overlap_end = min(iekf_end_time, vp_end_time)
    min_time = min(iekf_start_time, vp_start_time)
    
    print(f"IEKF time range: {iekf_start_time:.2f} to {iekf_end_time:.2f} seconds")
    print(f"Vehicle pose time range: {vp_start_time:.2f} to {vp_end_time:.2f} seconds")
    print(f"Overlap time range: {overlap_start:.2f} to {overlap_end:.2f} seconds")
    
    if overlap_start >= overlap_end:
        print("Error: No temporal overlap between IEKF and vehicle pose data")
        return
    
    # Filter IEKF data to overlap period
    iekf_mask = (iekf_time >= overlap_start) & (iekf_time <= overlap_end)
    iekf_time_overlap = iekf_time[iekf_mask]

    # Vehicle pose data
    vp_position = np.column_stack([
        vehicle_pose_extracted['position_x'].values,
        vehicle_pose_extracted['position_y'].values,
        vehicle_pose_extracted['position_z'].values
    ])
    
    vp_velocity = np.column_stack([
        vehicle_pose_extracted['velocity_V_x'].values,
        vehicle_pose_extracted['velocity_V_y'].values,
        vehicle_pose_extracted['velocity_V_z'].values
    ])
    
    vp_rotation_rates = np.column_stack([
        vehicle_pose_extracted['rotational_velocity_V_x'].values,
        vehicle_pose_extracted['rotational_velocity_V_y'].values,
        vehicle_pose_extracted['rotational_velocity_V_z'].values
    ])
    
    # Convert vehicle pose quaternions to rotation objects for SLERP
    vp_quaternions = np.column_stack([
        vehicle_pose_extracted['orientation_x'].values,
        vehicle_pose_extracted['orientation_y'].values,
        vehicle_pose_extracted['orientation_z'].values,
        vehicle_pose_extracted['orientation_w'].values
    ])
    
    vp_accel_rates = np.column_stack([
        vehicle_pose_extracted['acceleration_V_x'].values,
        vehicle_pose_extracted['acceleration_V_y'].values,
        vehicle_pose_extracted['acceleration_V_z'].values
    ])
    # Normalize quaternions
    quat_norms = np.linalg.norm(vp_quaternions, axis=1)
    vp_quaternions = vp_quaternions / quat_norms[:, np.newaxis]
    
    vp_rotations = R.from_quat(vp_quaternions)
    vp_euler = vp_rotations.as_euler('xyz', degrees=False)

    idx = np.searchsorted(vp_time_seconds, iekf_time_overlap[0], side='right') - 1

    def linear_interpolation(t1, t2, t_target, data1, data2):
        if t1 == t2:
            # If timestamps are identical, return average of data values
            return (data1 + data2) / 2.0
        
        # Linear interpolation formula: data = data1 + (data2 - data1) * (t_target - t1) / (t2 - t1)
        alpha = (t_target - t1) / (t2 - t1)
        return data1 + alpha * (data2 - data1)
    position_start = linear_interpolation(
        vp_time_seconds[idx], vp_time_seconds[idx + 1],
        iekf_time_overlap[0],
        vp_position[idx], vp_position[idx + 1])

    def slerp_interpolation(t1, t2, t_target, rot1, rot2):

        from scipy.spatial.transform import Rotation as R
        from scipy.spatial.transform import Slerp
        
        if t1 == t2:
            # If timestamps are identical, return the first rotation
            return rot1
        
        # Calculate interpolation parameter alpha
        alpha = (t_target - t1) / (t2 - t1)
        
        # Clamp alpha to [0, 1] for safety
        alpha = np.clip(alpha, 0.0, 1.0)
        
        # Create SLERP interpolator with normalized time range [0, 1]
        times = np.array([0.0, 1.0])
        rotations = R.concatenate([rot1, rot2])
        
        slerp = Slerp(times, rotations)
        
        # Interpolate at the target alpha
        interpolated_rotation = slerp(alpha)
        
        return interpolated_rotation
    rot_start = slerp_interpolation(
        vp_time_seconds[idx], vp_time_seconds[idx + 1],
        iekf_time_overlap[0],
        vp_rotations[idx], vp_rotations[idx + 1])

    
    # Extract corresponding IEKF data
    iekf_position_valid = iekf_results['p'][iekf_mask]
    iekf_velocity_valid = iekf_results['v'][iekf_mask] # This is in vehicle frame from IEKF
    iekf_euler_valid = iekf_results['euler_angles'][iekf_mask]
    
    # Extract IEKF corrected rotation rates if available
    if 'omega' in iekf_results:
        iekf_omega_valid = iekf_results['omega'][iekf_mask]
    else:
        iekf_omega_valid = None

    if 'accel' in iekf_results:
        iekf_accel_valid = iekf_results['accel'][iekf_mask]
    else:
        iekf_accel_valid = None

    
    # Apply transformations to all IEKF data
    # Transform positions: p_aligned = p_iekf + offset
    iekf_position_aligned = np.array([
        rot_start.apply(iekf_position_valid[i]) + position_start
        for i in range(len(iekf_position_valid))
    ])
    #iekf_position_aligned = R_vp_0.apply(iekf_position_valid) + position_offset
    
    # Transform rotations: R_aligned = R_transform * R_iekf
    iekf_rotations_original = R.from_euler('xyz', iekf_euler_valid)
    iekf_rotations_aligned = rot_start * iekf_rotations_original
    iekf_euler_aligned = iekf_rotations_aligned.as_euler('xyz', degrees=False)
    
    # For velocities - IEKF velocities are already in vehicle frame, just apply rotation alignment
    # iekf_velocity_aligned = np.array([
    #     R_transform.apply(iekf_velocity_valid[i])
    #     for i in range(len(iekf_velocity_valid))
    # ])
    
    print(f"Velocity frame: Both IEKF and Vehicle Pose velocities are in vehicle frame")
    print(f"IEKF velocity range (aligned vehicle frame): X=[{iekf_velocity_valid[:, 0].min():.2f}, {iekf_velocity_valid[:, 0].max():.2f}] m/s")
    
    # Create comprehensive comparison plots
    fig, axes = plt.subplots(5, 3, figsize=figsize)
    fig.suptitle('IEKF vs Vehicle Pose Comparison (SLERP + Rigid Transform Aligned)', fontsize=16, fontweight='bold')
    
    # Row 1: Position comparison (aligned)
    position_labels = ['X Position', 'Y Position', 'Z Position']
    position_units = ['m', 'm', 'm']

    iekf_plot_time = iekf_time_overlap - min_time
    vp_plot_time = vp_time_seconds - min_time
    
    for i in range(3):
        axes[0, i].plot(iekf_plot_time, iekf_position_aligned[:, i], 'b-', label='IEKF (Aligned)', linewidth=1.5, alpha=0.8)
        axes[0, i].plot(vp_plot_time, vp_position[:, i], 'r--', label='Vehicle Pose', linewidth=1.5, alpha=0.8)
        axes[0, i].set_title(f'{position_labels[i]} (Rigid Transform Aligned)', fontweight='bold')
        axes[0, i].set_ylabel(f'Position ({position_units[i]})')
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)
        
    
    # Row 2: Velocity comparison (both in vehicle frame)
    velocity_labels = ['X Velocity (Vehicle)', 'Y Velocity (Vehicle)', 'Z Velocity (Vehicle)']
    velocity_units = ['m/s', 'm/s', 'm/s']
    
    for i in range(3):
        axes[1, i].plot(iekf_plot_time, iekf_velocity_valid[:, i], 'b-', label='IEKF (Vehicle Frame)', linewidth=1.5, alpha=0.8)
        axes[1, i].plot(vp_plot_time, vp_velocity[:, i], 'r--', label='Vehicle Pose (Vehicle Frame)', linewidth=1.5, alpha=0.8)
        axes[1, i].set_title(f'{velocity_labels[i]}', fontweight='bold')
        axes[1, i].set_ylabel(f'Velocity ({velocity_units[i]})')
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)
        
    
    # Row 3: Rotation comparison (aligned)
    rotation_labels = ['Roll', 'Pitch', 'Yaw']
    
    for i in range(3):
        axes[2, i].plot(iekf_plot_time, np.degrees(iekf_euler_aligned[:, i]), 'b-', label='IEKF (Aligned)', linewidth=1.5, alpha=0.8)
        axes[2, i].plot(vp_plot_time, np.degrees(vp_euler[:, i]), 'r--', label='Vehicle Pose (SLERP)', linewidth=1.5, alpha=0.8)
        axes[2, i].set_title(f'{rotation_labels[i]} (Rigid Transform Aligned)', fontweight='bold')
        axes[2, i].set_ylabel('Angle (degrees)')
        axes[2, i].legend()
        axes[2, i].grid(True, alpha=0.3)
        
    
    # Row 4: Rotation rates comparison
    rotation_rate_labels = ['Roll Rate', 'Pitch Rate', 'Yaw Rate']
    
    for i in range(3):
        if iekf_omega_valid is not None:
            # Apply rotation transform to angular velocities
            iekf_omega_aligned = np.array([
                rot_start.apply(iekf_omega_valid[j])
                for j in range(len(iekf_omega_valid))
            ])
            axes[3, i].plot(iekf_plot_time, iekf_omega_aligned[:, i], 'b-', label='IEKF (Aligned)', linewidth=1.5, alpha=0.8)
        
        axes[3, i].plot(vp_plot_time, vp_rotation_rates[:, i], 'r--', label='Vehicle Pose', linewidth=1.5, alpha=0.8)
        axes[3, i].set_title(f'{rotation_rate_labels[i]}', fontweight='bold')
        axes[3, i].set_ylabel('Rate (rad/s)')
        axes[3, i].set_xlabel('Time (s)')
        axes[3, i].legend()
        axes[3, i].grid(True, alpha=0.3)

    # Row 4: Rotation rates comparison
    accel_rate_labels = ['gx', 'gy', 'gz']

    for i in range(3):
        if iekf_accel_valid is not None:
            axes[4, i].plot(iekf_plot_time, iekf_accel_valid[:, i], 'b-', label='IEKF (Aligned)', linewidth=1.5, alpha=0.8)

        axes[4, i].plot(vp_plot_time, vp_accel_rates[:, i], 'r--', label='Vehicle Pose', linewidth=1.5, alpha=0.8)
        axes[4, i].set_title(f'{accel_rate_labels[i]}', fontweight='bold')
        axes[4, i].set_ylabel('Rate (m/s2)')
        axes[4, i].set_xlabel('Time (s)')
        axes[4, i].legend()
        axes[4, i].grid(True, alpha=0.3)


    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"IEKF vs Vehicle Pose comparison plot saved to: {save_path}")
    
    plt.show()

def compute_imu_initial_stats(ego_motion_extracted, initial_duration=3.0, verbose=True):
    """
    Compute mean values of IMU acceleration and gyroscope data for the first few seconds.
    This is useful for initial bias estimation and sensor calibration.
    
    Args:
        ego_motion_extracted: DataFrame with ego motion data containing:
            - timestamp: System timestamp
            - imu_gx, imu_gy, imu_gz: IMU acceleration data (m/s²)
            - imu_roll, imu_pitch, imu_yaw: IMU rotation rate data (rad/s)
        initial_duration: Duration in seconds to analyze from the start (default: 3.0)
        verbose: Enable detailed output
        
    Returns:
        dict: Contains mean values and statistics:
            - mean_accel: Mean acceleration [gx, gy, gz] (m/s²)
            - mean_gyro: Mean gyro rates [roll, pitch, yaw] (rad/s)
            - std_accel: Standard deviation of acceleration
            - std_gyro: Standard deviation of gyro rates
            - num_samples: Number of samples used
            - duration: Actual duration of data used
            - sampling_rate: Estimated sampling rate
    """
    if ego_motion_extracted is None or ego_motion_extracted.empty:
        print("Error: No ego motion data provided")
        return None
    
    # Check if required columns exist
    required_columns = ['timestamp', 'imu_gx', 'imu_gy', 'imu_gz', 
                       'imu_roll', 'imu_pitch', 'imu_yaw']
    missing_columns = [col for col in required_columns if col not in ego_motion_extracted.columns]
    
    if missing_columns:
        print(f"Error: Missing columns: {missing_columns}")
        return None
    
    # Convert timestamps to seconds from start
    timestamps = ego_motion_extracted['timestamp'].values
    if timestamps.max() > 1e12:  # If in nanoseconds
        time_seconds = (timestamps - timestamps[0]) / 1e9
    else:  # If already in seconds
        time_seconds = timestamps - timestamps[0]
    
    # Find indices within the initial duration
    initial_mask = time_seconds <= initial_duration
    num_initial_samples = np.sum(initial_mask)
    
    if num_initial_samples == 0:
        print(f"Error: No samples found within the first {initial_duration} seconds")
        return None
    
    # Get actual duration of analyzed data
    actual_duration = time_seconds[initial_mask][-1] if num_initial_samples > 1 else 0
    
    # Extract IMU data for the initial period
    initial_data = ego_motion_extracted[initial_mask]
    
    # Compute mean values
    mean_accel = np.array([
        initial_data['imu_gx'].mean(),
        initial_data['imu_gy'].mean(), 
        initial_data['imu_gz'].mean()
    ])
    
    mean_gyro = np.array([
        initial_data['imu_roll'].mean(),
        initial_data['imu_pitch'].mean(),
        initial_data['imu_yaw'].mean()
    ])
    
    # Compute standard deviations
    std_accel = np.array([
        initial_data['imu_gx'].std(),
        initial_data['imu_gy'].std(),
        initial_data['imu_gz'].std()
    ])
    
    std_gyro = np.array([
        initial_data['imu_roll'].std(),
        initial_data['imu_pitch'].std(),
        initial_data['imu_yaw'].std()
    ])
    
    # Estimate sampling rate
    if num_initial_samples > 1:
        sampling_rate = (num_initial_samples - 1) / actual_duration
    else:
        sampling_rate = 0
    
    # Create results dictionary
    results = {
        'mean_accel': mean_accel,
        'mean_gyro': mean_gyro,
        'std_accel': std_accel,
        'std_gyro': std_gyro,
        'num_samples': num_initial_samples,
        'duration': actual_duration,
        'sampling_rate': sampling_rate,
        'initial_duration_requested': initial_duration
    }
    
    if verbose:
        print(f"\nIMU Initial Statistics (First {initial_duration} seconds):")
        print("=" * 60)
        print(f"Duration analyzed: {actual_duration:.3f} seconds")
        print(f"Number of samples: {num_initial_samples}")
        print(f"Estimated sampling rate: {sampling_rate:.1f} Hz")
        print()
        
        print("Acceleration Statistics:")
        print("-" * 30)
        accel_labels = ['X (gx)', 'Y (gy)', 'Z (gz)']
        for i, label in enumerate(accel_labels):
            print(f"  {label:<8}: Mean = {mean_accel[i]:8.4f} m/s², Std = {std_accel[i]:8.4f} m/s²")
        
        # Compare Z-axis with gravity
        gravity_error = abs(mean_accel[2]) - 9.80665
        print(f"  Z-axis vs gravity: {gravity_error:+.4f} m/s² (expected ~9.807 m/s²)")
        
        print()
        print("Gyroscope Statistics:")
        print("-" * 30)
        gyro_labels = ['Roll', 'Pitch', 'Yaw']
        for i, label in enumerate(gyro_labels):
            print(f"  {label:<8}: Mean = {mean_gyro[i]:8.6f} rad/s, Std = {std_gyro[i]:8.6f} rad/s")
            print(f"  {label:<8}: Mean = {np.degrees(mean_gyro[i]):8.4f} deg/s, Std = {np.degrees(std_gyro[i]):8.4f} deg/s")
        
        print()
        print("Sensor Quality Assessment:")
        print("-" * 30)
        
        # Check if gyroscope biases are reasonable (should be close to zero for stationary vehicle)
        gyro_magnitude = np.linalg.norm(mean_gyro)
        print(f"  Gyro bias magnitude: {gyro_magnitude:.6f} rad/s ({np.degrees(gyro_magnitude):.4f} deg/s)")
        
        if gyro_magnitude < 0.01:  # Less than 0.01 rad/s ≈ 0.57 deg/s
            print("  ✓ Gyro bias appears reasonable for stationary period")
        else:
            print("  ⚠ High gyro bias - vehicle may be moving or sensor needs calibration")
        
        # Check gyro noise levels
        gyro_noise_level = np.mean(std_gyro)
        print(f"  Average gyro noise: {gyro_noise_level:.6f} rad/s ({np.degrees(gyro_noise_level):.4f} deg/s)")
        
        if gyro_noise_level < 0.005:
            print("  ✓ Low gyro noise level")
        elif gyro_noise_level < 0.02:
            print("  ○ Moderate gyro noise level")
        else:
            print("  ⚠ High gyro noise level")
        
        # Check accelerometer alignment with gravity
        accel_magnitude = np.linalg.norm(mean_accel)
        print(f"  Accel magnitude: {accel_magnitude:.4f} m/s² (expected ~9.807 m/s²)")
        
        gravity_error_pct = abs(gravity_error) / 9.80665 * 100
        print(f"  Gravity alignment error: {gravity_error_pct:.2f}%")
        
        if gravity_error_pct < 2.0:
            print("  ✓ Good gravity alignment")
        elif gravity_error_pct < 5.0:
            print("  ○ Reasonable gravity alignment") 
        else:
            print("  ⚠ Poor gravity alignment - check sensor mounting")
        
        # Check if vehicle appears stationary
        lateral_accel = np.sqrt(mean_accel[0]**2 + mean_accel[1]**2)
        print(f"  Lateral acceleration: {lateral_accel:.4f} m/s²")
        
        if lateral_accel < 0.5 and gyro_magnitude < 0.01:
            print("  ✓ Vehicle appears stationary - good for bias estimation")
        else:
            print("  ⚠ Vehicle may be moving - bias estimates may be affected")
    
    return results

def plot_frame_time_differences(ego_motion_extracted, save_path=None, figsize=(12, 6)):
    """
    Simple plot of time differences between consecutive frames for ego motion data.
    
    Args:
        ego_motion_extracted: DataFrame containing ego motion data with timestamp column
        save_path: Optional path to save the plot
        figsize: Figure size (width, height) in inches
    """
    if ego_motion_extracted is None or ego_motion_extracted.empty:
        print("Error: No ego motion data provided")
        return
    
    # Check if timestamp column exists
    if 'timestamp' not in ego_motion_extracted.columns:
        print("Error: timestamp column not found in ego motion data")
        return
    
    # Convert timestamps to seconds from start
    timestamps = ego_motion_extracted['timestamp'].values
    if timestamps.max() > 1e12:  # If in nanoseconds
        time_seconds = (timestamps - timestamps[0]) / 1e9
    else:  # If already in seconds
        time_seconds = timestamps - timestamps[0]
    
    # Calculate time differences between consecutive frames
    time_diffs = np.diff(time_seconds)
    time_midpoints = time_seconds[:-1] + time_diffs / 2  # Midpoint times for plotting
    
    # Create simple plot
    plt.figure(figsize=figsize)
    plt.plot(time_midpoints, time_diffs * 1000, 'b-', linewidth=0.8, alpha=0.8)
    plt.title('Time Difference Between Consecutive Frames', fontweight='bold', fontsize=14)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Time Difference (ms)')
    plt.grid(True, alpha=0.3)
    plt.xlim(time_seconds.min(), time_seconds.max())
    
    # Save the plot if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Frame time differences plot saved to: {save_path}")
    
    plt.show()

def extract_data_from_parquet(input_dir, output_dir, verbose=False, force=False):
    """
    Extract data from parquet files or load existing extracted data.
    
    Args:
        input_dir: Directory containing input parquet files
        output_dir: Directory to save/load extracted data
        verbose: Enable verbose output
        force: If True, force re-extraction; if False, load existing if available
    
    Returns:
        tuple: (ego_motion_extracted, ublox_extracted)
    """
    # Convert input directory to Path object
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Define paths for extracted files
    ego_extracted_file = output_dir / "ego_motion_extracted.parquet"
    ublox_extracted_file = output_dir / "ublox_extracted.parquet"
    vehicle_pose_extracted_file = output_dir / "vehicle_pose_extracted.parquet"
    
    # Check if extracted files exist and force is False
    if not force and ego_extracted_file.exists() and ublox_extracted_file.exists() and vehicle_pose_extracted_file.exists():
        print("Found existing extracted parquet files. Loading from cache...")
        print("=" * 60)
        
        try:
            # Load existing extracted data
            print(f"Loading ego motion data from: {ego_extracted_file}")
            ego_motion_extracted = pd.read_parquet(ego_extracted_file)
            
            print(f"Loading ublox data from: {ublox_extracted_file}")
            ublox_extracted = pd.read_parquet(ublox_extracted_file)

            print(f"Loading vehicle pose data from: {vehicle_pose_extracted_file}")
            vehicle_pose_extracted = pd.read_parquet(vehicle_pose_extracted_file)
            
            print(f"Successfully loaded extracted data from cache")
            print(f"Ego motion shape: {ego_motion_extracted.shape}")
            print(f"Ublox shape: {ublox_extracted.shape}")
            print(f"Vehicle pose shape: {vehicle_pose_extracted.shape}")

            # Display sample data if verbose mode is enabled
            if verbose:
                print("\nSample cached ego motion data:")
                with pd.option_context('display.max_columns', None, 'display.width', None):
                    print(ego_motion_extracted.head())
                
                print("\nSample cached ublox data:")
                with pd.option_context('display.max_columns', None, 'display.width', None):
                    print(ublox_extracted.head())

                print("\nSample cached vehicle pose data:")
                with pd.option_context('display.max_columns', None, 'display.width', None):
                    print(vehicle_pose_extracted.head())

            return ego_motion_extracted, ublox_extracted, vehicle_pose_extracted

        except Exception as e:
            print(f"Error loading cached data: {str(e)}")
            print("Falling back to re-extraction...")
            force = True  # Force re-extraction if loading fails
    
    elif not force and (ego_extracted_file.exists() or ublox_extracted_file.exists()):
        print("Warning: Only partial extracted data found. Re-extracting all data...")
        force = True
    
    if force and ego_extracted_file.exists():
        print(f"Force mode: Overwriting existing file {ego_extracted_file}")
    
    if force and ublox_extracted_file.exists():
        print(f"Force mode: Overwriting existing file {ublox_extracted_file}")

    if force and vehicle_pose_extracted_file.exists():  
        print(f"Force mode: Overwriting existing file {vehicle_pose_extracted_file}")
    
    # Check if input directory exists
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return None, None
    
    if not input_dir.is_dir():
        print(f"Error: Input path is not a directory: {input_dir}")
        return None, None

    # Define file paths within the input directory
    file1_path = input_dir / "all_speed_ego_motion_perception_inputs.parquet"
    file2_path = input_dir / "ublox_all_speed_egomotion_perception_inputs.parquet"
    file3_path = input_dir / "vehicle_pose.parquet"
    
    # Check if files exist
    if not file1_path.exists():
        print(f"Error: File not found: {file1_path}")
        return None, None
    
    if not file2_path.exists():
        print(f"Error: File not found: {file2_path}")
        return None, None

    if not file3_path.exists():
        print(f"Error: File not found: {file3_path}")
        return None, None

    print(f"Extracting data from parquet files in directory: {input_dir}")
    print("="*60)
    
    # Read the first file (ego motion perception inputs)
    print("Reading all_speed_ego_motion_perception_inputs.parquet...")
    ego_motion_df = read_parquet_file(file1_path, 'all_speed_ego_motion_perception_inputs')
    
    # Read the second file (ublox GPS data)
    print("Reading ublox_all_speed_egomotion_perception_inputs.parquet...")
    ublox_df = read_parquet_file(file2_path, 'ublox_all_speed_egomotion_perception_inputs')

    # Read the third file (vehicle pose data)
    print("Reading vehicle_pose.parquet...")
    vehicle_pose_df = read_parquet_file(file3_path, 'vehicle_pose')
    
    # Extract relevant data
    print("\nExtracting relevant data...")
    print("-" * 40)
    
    ego_motion_extracted = extract_ego_motion_data(ego_motion_df)
    ublox_extracted = extract_ublox_data(ublox_df)
    vehicle_pose_extracted = extract_vehicle_pose_data(vehicle_pose_df)

    # Save extracted data
    print("\nSaving extracted data...")
    print("-" * 40)
    save_extracted_data(ego_motion_extracted, ublox_extracted, vehicle_pose_extracted, ego_extracted_file, ublox_extracted_file, vehicle_pose_extracted_file)

    # Display sample data if verbose mode is enabled
    if verbose:
        if ego_motion_extracted is not None:
            print("\nSample extracted ego motion data:")
            with pd.option_context('display.max_columns', None, 'display.width', None):
                print(ego_motion_extracted.head())
        
        if ublox_extracted is not None:
            print("\nSample extracted ublox data:")
            with pd.option_context('display.max_columns', None, 'display.width', None):
                print(ublox_extracted.head())

        if vehicle_pose_extracted is not None:
            print("\nSample extracted vehicle pose data:")
            with pd.option_context('display.max_columns', None, 'display.width', None):
                print(vehicle_pose_extracted.head())
    
    # Summary statistics
    if ego_motion_extracted is not None:
        print(f"\nExtracted ego motion data summary:")
        print(f"  - Total rows: {len(ego_motion_extracted)}")
        print(f"  - IMU gyro range (gx): {ego_motion_extracted['imu_gx'].min():.3f} to {ego_motion_extracted['imu_gx'].max():.3f}")
        print(f"  - IMU gyro range (gy): {ego_motion_extracted['imu_gy'].min():.3f} to {ego_motion_extracted['imu_gy'].max():.3f}")
        print(f"  - IMU gyro range (gz): {ego_motion_extracted['imu_gz'].min():.3f} to {ego_motion_extracted['imu_gz'].max():.3f}")
        print(f"  - Wheel speed range: {ego_motion_extracted[['wheel_vxfr', 'wheel_vxfl', 'wheel_vxrr', 'wheel_vxrl']].min().min():.3f} to {ego_motion_extracted[['wheel_vxfr', 'wheel_vxfl', 'wheel_vxrr', 'wheel_vxrl']].max().max():.3f}")
    
    if ublox_extracted is not None:
        print(f"\nExtracted ublox data summary:")
        print(f"  - Total rows: {len(ublox_extracted)}")
        print(f"  - Latitude range: {ublox_extracted['latitude'].min():.6f} to {ublox_extracted['latitude'].max():.6f}")
        print(f"  - Longitude range: {ublox_extracted['longitude'].min():.6f} to {ublox_extracted['longitude'].max():.6f}")
        print(f"  - Height range: {ublox_extracted['height'].min()} to {ublox_extracted['height'].max()}")
        print(f"  - Fix types distribution:")
        fix_counts = ublox_extracted['fix_type'].value_counts().sort_index()
        for fix_type, count in fix_counts.items():
            print(f"    Fix type {fix_type}: {count} samples ({count/len(ublox_extracted)*100:.1f}%)")

    if ego_motion_extracted is not None:
        print("\nGenerating IMU time series plots...")
        print("-" * 40)

        compute_imu_initial_stats(ego_motion_extracted, initial_duration=3.0, verbose=verbose)

        plot_frame_time_differences(ego_motion_extracted, save_path=Path(output_dir) / "frame_time_differences.png")
        
        # Plot IMU data
        imu_plot_path = Path(output_dir) / "imu_time_series.png"
        plot_imu_time_series(ego_motion_extracted, vehicle_pose_extracted, save_path=imu_plot_path)

        # Plot wheel speed data
        wheel_plot_path = Path(output_dir) / "wheel_speed_time_series.png"
        plot_wheel_speed_time_series(ego_motion_extracted, save_path=wheel_plot_path)

    return ego_motion_extracted, ublox_extracted, vehicle_pose_extracted


def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Read and process parquet files containing IMU and GPS data"
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing the parquet files"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./extracted_data",
        help="Directory to save extracted data (default: ./extracted_data)"
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Force re-extraction of data even if output files exist"
    )
    parser.add_argument(
        "--run-vehicle-pose",
        action="store_true",
        help="Force re-extraction of data even if output files exist"
    )
    
    return parser.parse_args()

def main():
    """
    Main function to read both parquet files from the specified directory.
    """
    # Parse command-line arguments
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)
    ego_motion_extracted, ublox_extracted, vehicle_pose_extracted = extract_data_from_parquet(args.input_dir, args.output_dir, args.verbose, args.force_extract)

    if ego_motion_extracted is None or ublox_extracted is None:
        print("Error: Data extraction failed. Exiting.")
        sys.exit(1)

    scaling_results = analyze_speed_scaling(ego_motion_extracted, None, vehicle_pose_extracted, args.output_dir, args.force_extract)
    if scaling_results is None:
        print("Error: Speed scaling analysis failed. Exiting.")
        sys.exit(1)

    wheel_speed_scaling = scaling_results.get('p50_scaling', 1.0)

    # Run IEKF after extracting ego motion data
    print("\nRunning IEKF with ego motion data...")
    print("=" * 50)

    # Run IEKF with ego motion data
    if args.run_vehicle_pose:
        print("\nRunning IEKF with vehicle pose data...")
        iekf_results = run_iekf_with_vehicle_pose(
            vehicle_pose_extracted,
            wheel_speed_scaling,
            output_dir=args.output_dir,
            verbose=args.verbose
        )

    else:
        print("\nRunning IEKF with ego motion data...")     
        iekf_results = run_iekf_with_ego_motion(
            ego_motion_extracted,
            wheel_speed_scaling,
            output_dir=args.output_dir,
            verbose=args.verbose
        )
    if iekf_results is not None:
        # Plot IEKF results
        iekf_plot_path = Path(args.output_dir) / "iekf_results.png"
        plot_iekf_results(iekf_results, save_path=iekf_plot_path)
        plot_iekf_vs_vehicle_pose(iekf_results, vehicle_pose_extracted,  save_path=Path(args.output_dir) / "iekf_vs_vehicle_pose.png")

if __name__ == "__main__":
    main()