"""
NumPy implementation of the Invariant Extended Kalman Filter (IEKF) for dead reckoning.

This module provides a NumPy-based implementation of the Invariant Extended Kalman Filter
for state estimation in the AI-IMU Dead Reckoning system. The IEKF integrates IMU measurements
(acceleration and angular velocity) with zero lateral and vertical velocity constraints
to estimate vehicle position, velocity, and orientation.

The filter estimates the following state variables:
- Rotation matrix (3x3 orientation)
- Velocity (3D vector)
- Position (3D vector)
- Gyroscope bias (3D vector)
- Accelerometer bias (3D vector)
- Car-to-IMU rotation (3x3 orientation)
- Car-to-IMU translation (3D vector)

This implementation is primarily used for evaluation purposes. For training,
see the PyTorch implementation in utils_torch_filter.py.
"""

import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=2)
import scipy.linalg
from termcolor import cprint
from utils import *
from tqdm import tqdm

class NUMPYIEKF:
    """
    NumPy implementation of the Invariant Extended Kalman Filter (IEKF).

    This class implements an Invariant Extended Kalman Filter for vehicle state estimation
    using IMU measurements. The filter maintains and updates the state variables (orientation,
    position, velocity, biases) and their uncertainties represented as covariance matrices.

    The implementation follows the invariant Kalman filter framework, where error states
    are represented in the Lie algebra of the state space. This approach ensures that
    orientation errors maintain proper structure on the manifold of rotation matrices.

    The filter runs in two phases:
    1. Propagation: uses IMU measurements to predict the next state
    2. Update: incorporates zero velocity constraints to correct the state

    Attributes:
        Id2 (numpy.ndarray): 2x2 identity matrix
        Id3 (numpy.ndarray): 3x3 identity matrix
        Id6 (numpy.ndarray): 6x6 identity matrix
        IdP (numpy.ndarray): 21x21 identity matrix (state dimension)
    """
    Id2 = np.eye(2)
    Id3 = np.eye(3)
    Id6 = np.eye(6)
    IdP = np.eye(21)

    def __init__(self, parameter_class=None, verbose=False):
        """
        Initialize the IEKF with parameters.

        Creates an IEKF instance and sets up the filter parameters. If no parameter
        class is provided, uses the default Parameters inner class.

        Args:
            parameter_class (class, optional): A class that provides filter parameters.
                                              Must have the same attributes as NUMPYIEKF.Parameters.
                                              If None, uses the default Parameters.

        Note:
            The following attributes are initialized from the parameter class:
            - g: gravity vector
            - cov_*: Process noise covariance parameters
            - cov_*0: Initial state covariance parameters
            - Q: Process noise covariance matrix
            - Q_dim: Process noise dimension
            - P_dim: State covariance dimension
            - n_normalize_rot: Frequency of rotation normalization
            - n_normalize_rot_c_i: Frequency of car-IMU rotation normalization
        """
        # variables to initialize with `filter_parameters`
        self.g = None
        self.cov_omega = None
        self.cov_acc = None
        self.cov_b_omega = None
        self.cov_b_acc = None
        self.cov_Rot_c_i = None
        self.cov_t_c_i = None
        self.cov_lat = None
        self.cov_up = None
        self.cov_b_omega0 = None
        self.cov_b_acc0 = None
        self.cov_Rot0 = None
        self.cov_v0 = None
        self.cov_Rot_c_i0 = None
        self.cov_t_c_i0 = None
        self.Q = None
        self.Q_dim = None
        self.n_normalize_rot = None
        self.n_normalize_rot_c_i = None
        self.P_dim = None
        self.verbose = verbose

        # set the parameters
        if parameter_class is None:
            filter_parameters = NUMPYIEKF.Parameters()
        else:
            filter_parameters = parameter_class()
        self.filter_parameters = filter_parameters
        self.set_param_attr()

    class Parameters:
        g = np.array([0, 0, -9.80665])
        """gravity vector"""

        P_dim = 21
        """covariance dimension"""

        Q_dim = 18
        """process noise covariance dimension"""

        # Process noise covariance
        cov_omega = 1e-3
        """gyro covariance"""
        cov_acc = 1e-2
        """accelerometer covariance"""
        cov_b_omega = 6e-9
        """gyro bias covariance"""
        cov_b_acc = 2e-4
        """accelerometer bias covariance"""
        cov_Rot_c_i = 1e-9
        """car to IMU orientation covariance"""
        cov_t_c_i = 1e-9
        """car to IMU translation covariance"""

        cov_lat = 0.2
        """Zero lateral velocity covariance"""
        cov_up = 300
        """Zero lateral velocity covariance"""

        cov_Rot0 = 1e-3
        """initial pitch and roll covariance"""
        cov_b_omega0 = 6e-3
        """initial gyro bias covariance"""
        cov_b_acc0 = 4e-3
        """initial accelerometer bias covariance"""
        cov_v0 = 1e-1
        """initial velocity covariance"""
        cov_Rot_c_i0 = 1e-6
        """initial car to IMU pitch and roll covariance"""
        cov_t_c_i0 = 5e-3
        """initial car to IMU translation covariance"""

        # numerical parameters
        n_normalize_rot = 100
        """timestamp before normalizing orientation"""
        n_normalize_rot_c_i = 1000
        """timestamp before normalizing car to IMU orientation"""

        def __init__(self, **kwargs):
            self.set(**kwargs)

        def set(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    def set_param_attr(self):
        """
        Copy attributes from the filter_parameters object to this instance.

        Transfers all non-callable attributes from the filter_parameters object
        to the IEKF instance and sets up the process noise covariance matrix Q.

        The Q matrix is constructed as a diagonal matrix with the following structure:
        - Gyroscope noise (3x3): cov_omega
        - Accelerometer noise (3x3): cov_acc
        - Gyroscope bias drift (3x3): cov_b_omega
        - Accelerometer bias drift (3x3): cov_b_acc
        - Car-to-IMU rotation drift (3x3): cov_Rot_c_i
        - Car-to-IMU translation drift (3x3): cov_t_c_i
        """
        # get a list of attribute only
        attr_list = [a for a in dir(self.filter_parameters) if not a.startswith('__')
                     and not callable(getattr(self.filter_parameters, a))]
        for attr in attr_list:
            setattr(self, attr, getattr(self.filter_parameters, attr))

        # Set up the process noise covariance matrix
        self.Q = np.diag([self.cov_omega, self.cov_omega, self. cov_omega,
                           self.cov_acc, self.cov_acc, self.cov_acc,
                           self.cov_b_omega, self.cov_b_omega, self.cov_b_omega,
                           self.cov_b_acc, self.cov_b_acc, self.cov_b_acc,
                           self.cov_Rot_c_i, self.cov_Rot_c_i, self.cov_Rot_c_i,
                           self.cov_t_c_i, self.cov_t_c_i, self.cov_t_c_i])

    def run(self, t, u, measurements_covs, v_mes, p_mes0, N, ang0, bias0=None, p4h_params=None):
        """
        Run the IEKF over a sequence of IMU measurements.

        This is the main method that processes a sequence of IMU measurements and estimates
        the vehicle state at each timestamp.

        Args:
            t (numpy.ndarray): Timestamps (seconds)
            u (numpy.ndarray): IMU measurements [angular_velocity (3), acceleration (3)]
            measurements_covs (numpy.ndarray): Measurement covariances for zero velocity constraints
            v_mes (numpy.ndarray):  velocity measurements
            p_mes0 (numpy.ndarray): Initial position measurement
            N (int, optional): Number of time steps to process. If None, uses all available data.
            ang0 (numpy.ndarray): Initial orientation as [roll, pitch, yaw] in radians
            bias0 (numpy.ndarray, optional): Initial bias estimates [gyro_bias (3), accel_bias (3)]

        Returns:
            tuple:
                - Rot (numpy.ndarray): Array of orientation matrices (N, 3, 3)
                - v (numpy.ndarray): Array of velocity vectors (N, 3)
                - p (numpy.ndarray): Array of position vectors (N, 3)
                - b_omega (numpy.ndarray): Array of gyroscope bias estimates (N, 3)
                - b_acc (numpy.ndarray): Array of accelerometer bias estimates (N, 3)
                - Rot_c_i (numpy.ndarray): Array of car-to-IMU rotation matrices (N, 3, 3)
                - t_c_i (numpy.ndarray): Array of car-to-IMU translation vectors (N, 3)
        """
        dt = t[1:] - t[:-1]  # (s)
        if N is None:
            N = u.shape[0]
        if p4h_params is not None:
            Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P = self.init_p4h_run(dt, u, p_mes0, v_mes,
                                       ang0, N, p4h_params)
            print(f"Init P from p4h params: {np.diag(P)}")
        else:
            Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P = self.init_run(p_mes0, v_mes[0],
                                        ang0, N, bias0=bias0)

        for i in tqdm(range(1, N)):
            Rot[i], v[i], p[i], b_omega[i], b_acc[i], Rot_c_i[i], t_c_i[i], P = \
                self.propagate(Rot[i-1], v[i-1], p[i-1], b_omega[i-1], b_acc[i-1], Rot_c_i[i-1],
                               t_c_i[i-1], P, u[i], dt[i-1])

            # Update step - correct the state using zero velocity constraints
            Rot[i], v[i], p[i], b_omega[i], b_acc[i], Rot_c_i[i], t_c_i[i], P = \
                self.update(Rot[i], v[i], p[i], b_omega[i], b_acc[i], Rot_c_i[i], t_c_i[i], P, u[i],
                            i, measurements_covs[i], v_mes[i])

            # Correct numerical error every n_normalize_rot steps
            if i % self.n_normalize_rot == 0:
                Rot[i] = self.normalize_rot(Rot[i])

            # Correct numerical error every n_normalize_rot_c_i steps
            if i % self.n_normalize_rot_c_i == 0:
                Rot_c_i[i] = self.normalize_rot(Rot_c_i[i])

        return Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i

    def init_p4h_run(self, dt, u, p_mes, v_mes, ang0, N, p4h_params):
        Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i = self.init_p4h_saved_state(dt, N, ang0, p4h_params)
        Rot[0] = self.from_rpy(ang0[0], ang0[1], ang0[2])
        v[0] = v_mes[0]
        P = self.init_covariance()
        return Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P

    def init_p4h_saved_state(self, dt, N, ang0, p4h_params):
        Rot = np.zeros((N, 3, 3))
        v = np.zeros((N, 3))
        p = np.zeros((N, 3))
        b_omega = np.zeros((N, 3))
        b_acc = np.zeros((N, 3))
        Rot_c_i = np.zeros((N, 3, 3))
        t_c_i = np.zeros((N, 3))

        Rot[0] = p4h_params['R_LI']
        v[0] = p4h_params['v_LI_L']
        p[0] = p4h_params['p_LI_L']

        T_IV = np.linalg.inv(p4h_params['T_VI'])

        Rot_c_i[0] = T_IV[:3, :3]
        t_c_i[0] = T_IV[:3, 3]
        return Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i

    def init_run(self, p_mes0, v_mes0, ang0, N, bias0=None):
        """
        Initialize the filter run with starting values.

        Sets up the initial state for the filter run and initializes the covariance matrix.

        Args:
            p_mes0 (numpy.ndarray): Initial position measurement
            v_mes0 (numpy.ndarray): Initial velocity measurement
            ang0 (numpy.ndarray): Initial orientation as [roll, pitch, yaw]
            N (int): Number of time steps
            bias0 (numpy.ndarray, optional): Initial bias estimates [gyro_bias (3), accel_bias (3)]

        Returns:
            tuple:
                - Rot (numpy.ndarray): Array of orientation matrices
                - v (numpy.ndarray): Array of velocity vectors
                - p (numpy.ndarray): Array of position vectors
                - b_omega (numpy.ndarray): Array of gyroscope bias estimates
                - b_acc (numpy.ndarray): Array of accelerometer bias estimates
                - Rot_c_i (numpy.ndarray): Array of car-to-IMU rotation matrices
                - t_c_i (numpy.ndarray): Array of car-to-IMU translation vectors
                - P (numpy.ndarray): Initial covariance matrix
        """
        # Initialize state arrays
        Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i = self.init_saved_state(N, p_mes0, v_mes0, ang0, bias0=bias0)

        # Initialize covariance matrix
        P = self.init_covariance()

        return Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P

    def init_covariance(self):
        """
        Initialize the state covariance matrix.

        Creates the initial covariance matrix P with appropriate values for each state component.
        The covariance matrix is structured as follows:
        - Orientation error (indices 0-2)
        - Velocity error (indices 3-5)
        - Position error (indices 6-8)
        - Gyroscope bias error (indices 9-11)
        - Accelerometer bias error (indices 12-14)
        - Car-to-IMU rotation error (indices 15-17)
        - Car-to-IMU translation error (indices 18-20)

        Note that initial yaw error is not included (index 2) as it's unobservable.

        Returns:
            numpy.ndarray: Initial covariance matrix (P_dim x P_dim)
        """
        P = np.zeros((self.P_dim, self.P_dim))
        P[:2, :2] = self.cov_Rot0*self.Id2  # no yaw error
        P[3:5, 3:5] = self.cov_v0*self.Id2
        P[9:12, 9:12] = self.cov_b_omega0*self.Id3
        P[12:15, 12:15] = self.cov_b_acc0*self.Id3
        P[15:18, 15:18] = self.cov_Rot_c_i0*self.Id3
        P[18:21, 18:21] = self.cov_t_c_i0*self.Id3
        return P

    def init_saved_state(self, N, pos0, vel0, ang0, bias0=None):
        """
        Initialize arrays to store state variables during the filter run.

        Creates zeroed arrays for all state components with the appropriate dimensions.

        Args:
            N (int): Number of time steps
            pos0 (numpy.ndarray): Initial position as [x, y, z]
            vel0 (numpy.ndarray): Initial velocity as [vx, vy, vz]
            ang0 (numpy.ndarray): Initial orientation as [roll, pitch, yaw]
            bias0 (numpy.ndarray, optional): Initial bias estimates [gyro_bias (3), accel_bias (3)]

        Returns:
            tuple:
                - Rot (numpy.ndarray): Array of orientation matrices (N, 3, 3)
                - v (numpy.ndarray): Array of velocity vectors (N, 3)
                - p (numpy.ndarray): Array of position vectors (N, 3)
                - b_omega (numpy.ndarray): Array of gyroscope bias estimates (N, 3)
                - b_acc (numpy.ndarray): Array of accelerometer bias estimates (N, 3)
                - Rot_c_i (numpy.ndarray): Array of car-to-IMU rotation matrices (N, 3, 3)
                - t_c_i (numpy.ndarray): Array of car-to-IMU translation vectors (N, 3)
        """
        # Initialize state arrays with zeros
        Rot = np.zeros((N, 3, 3))
        v = np.zeros((N, 3))
        p = np.zeros((N, 3))
        b_omega = np.zeros((N, 3))
        b_acc = np.zeros((N, 3))
        Rot_c_i = np.zeros((N, 3, 3))
        t_c_i = np.zeros((N, 3))

        # Initialize car-to-IMU rotation as identity
        Rot_c_i[0] = np.eye(3)

        # Set initial orientation from roll, pitch, yaw
        Rot[0] = self.from_rpy(ang0[0], ang0[1], ang0[2])

        # Set initial velocity
        v[0] = vel0

        # Set initial position
        p[0] = pos0

        if bias0 is not None:
            b_omega[0] = bias0[:3]
            b_acc[0] = bias0[3:6]

        return Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i
    
    def reset_imu_bias_with_zupt(self, Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P, u, v_mes,
                             zupt_duration_threshold=2.0, velocity_threshold=0.1, 
                             bias_reset_factor=0.9, verbose=False):
        """
        Reset IMU bias estimates using Zero Velocity Update (ZUPT) when vehicle is stationary.
        
        This function detects when the vehicle has been stationary for a sufficient duration
        and uses this information to reset the IMU bias estimates. When the vehicle is truly
        stationary, any measured acceleration (after removing gravity) and angular velocity
        should be due to sensor bias and noise.
        
        Args:
            Rot (numpy.ndarray): Current orientation matrix (3x3)
            v (numpy.ndarray): Current velocity vector (3)
            p (numpy.ndarray): Current position vector (3)
            b_omega (numpy.ndarray): Current gyroscope bias (3)
            b_acc (numpy.ndarray): Current accelerometer bias (3)
            Rot_c_i (numpy.ndarray): Current car-to-IMU rotation matrix (3x3)
            t_c_i (numpy.ndarray): Current car-to-IMU translation vector (3)
            P (numpy.ndarray): Current state covariance matrix
            u (numpy.ndarray): Current IMU measurement [angular_velocity (3), acceleration (3)]
            zupt_duration_threshold (float): Minimum duration (seconds) of stationary period 
                                        required before bias reset
            velocity_threshold (float): Velocity magnitude threshold to consider vehicle stationary
            bias_reset_factor (float): Factor to blend old bias with new estimate (0.0=full reset, 1.0=no change)
            verbose (bool): Enable verbose logging
            
        Returns:
            tuple:
                - b_omega_reset (numpy.ndarray): Updated gyroscope bias (3)
                - b_acc_reset (numpy.ndarray): Updated accelerometer bias (3)
                - P_reset (numpy.ndarray): Updated covariance matrix with reset bias uncertainties
                - bias_reset_flag (bool): True if bias was actually reset
        """
        
        # Check if vehicle is stationary
        velocity_magnitude = np.linalg.norm(v_mes)
        is_stationary = velocity_magnitude < velocity_threshold
        
        if not is_stationary:
            if verbose:
                print(f"[BIAS_RESET] Vehicle moving (speed: {velocity_magnitude:.3f} m/s), no bias reset")
            return b_omega, b_acc, P, False
        
        # Initialize stationary tracking if not exists
        if not hasattr(self, '_stationary_start_time'):
            self._stationary_start_time = None
            self._stationary_omega_samples = []
            self._stationary_acc_samples = []
            self._current_time = 0.0
        
        # Update current time (this should be passed from outside in real implementation)
        self._current_time += 0.01  # Assume 100Hz for now - in real use, pass actual dt
        
        if self._stationary_start_time is None:
            # Start of stationary period
            self._stationary_start_time = self._current_time
            self._stationary_omega_samples = []
            self._stationary_acc_samples = []
            if verbose:
                print(f"[BIAS_RESET] Started stationary period at time {self._current_time:.2f}s")
        
        # Accumulate samples during stationary period
        self._stationary_omega_samples.append(u[:3].copy())
        
        # For accelerometer, we need to remove gravity component
        # Convert gravity to IMU frame
        gravity_imu = Rot.T.dot(self.g)
        acc_no_gravity = u[3:6] + gravity_imu
        self._stationary_acc_samples.append(acc_no_gravity.copy())
        
        # Check if we've been stationary long enough
        stationary_duration = self._current_time - self._stationary_start_time
        
        if stationary_duration < zupt_duration_threshold:
            if verbose:
                print(f"[BIAS_RESET] Stationary for {stationary_duration:.2f}s, need {zupt_duration_threshold:.2f}s")
            return b_omega, b_acc, P, False
        
        # We've been stationary long enough - estimate bias
        omega_samples = np.array(self._stationary_omega_samples)
        acc_samples = np.array(self._stationary_acc_samples)
        
        # Estimate bias as mean of samples during stationary period
        omega_bias_estimate = np.mean(omega_samples, axis=0)
        acc_bias_estimate = np.mean(acc_samples, axis=0)
        
        # Compute standard deviation to assess quality of estimate
        omega_std = np.std(omega_samples, axis=0)
        acc_std = np.std(acc_samples, axis=0)
        
        if verbose:
            print(f"[BIAS_RESET] Stationary period: {stationary_duration:.2f}s, {len(omega_samples)} samples")
            print(f"[BIAS_RESET] Current gyro bias: {b_omega}")
            print(f"[BIAS_RESET] Estimated gyro bias: {omega_bias_estimate} (std: {omega_std})")
            print(f"[BIAS_RESET] Current acc bias: {b_acc}")
            print(f"[BIAS_RESET] Estimated acc bias: {acc_bias_estimate} (std: {acc_std})")
        
        # Quality check - reject estimate if standard deviation is too high
        max_omega_std = 0.01  # rad/s
        max_acc_std = 0.2    # m/s²
        
        if np.any(omega_std > max_omega_std) or np.any(acc_std > max_acc_std):
            if verbose:
                print(f"[BIAS_RESET] WARNING: High noise during stationary period, skipping bias reset")
                print(f"[BIAS_RESET] Omega std: {omega_std} (max: {max_omega_std})")
                print(f"[BIAS_RESET] Acc std: {acc_std} (max: {max_acc_std})")
            return b_omega, b_acc, P, False
        
        # Blend old bias with new estimate
        b_omega_reset = bias_reset_factor * b_omega + (1 - bias_reset_factor) * omega_bias_estimate
        b_acc_reset = bias_reset_factor * b_acc + (1 - bias_reset_factor) * acc_bias_estimate
        
        # Update covariance matrix - reduce uncertainty in bias estimates
        P_reset = P.copy()
        
        # Reset bias covariance to smaller values since we have a good estimate
        bias_reset_cov_omega = self.cov_b_omega0 * 1.0 # 10x more confident
        bias_reset_cov_acc = self.cov_b_acc0 * 1.0      # 10x more confident
        
        P_reset[9:12, 9:12] = bias_reset_cov_omega * self.Id3   # Gyro bias covariance
        P_reset[12:15, 12:15] = bias_reset_cov_acc * self.Id3   # Acc bias covariance
        
        # Cross-correlations with bias should also be reduced
        P_reset[9:12, :9] *= 0.5    # Reduce correlation between gyro bias and pose
        P_reset[:9, 9:12] *= 0.5
        P_reset[12:15, :12] *= 0.5  # Reduce correlation between acc bias and pose/gyro bias  
        P_reset[:12, 12:15] *= 0.5
        
        if verbose:
            bias_omega_change = np.linalg.norm(b_omega_reset - b_omega)
            bias_acc_change = np.linalg.norm(b_acc_reset - b_acc)
            print(f"[BIAS_RESET] Gyro bias change: {bias_omega_change:.6f} rad/s")
            print(f"[BIAS_RESET] Acc bias change: {bias_acc_change:.6f} m/s²")
            print(f"[BIAS_RESET] Updated gyro bias: {b_omega_reset}")
            print(f"[BIAS_RESET] Updated acc bias: {b_acc_reset}")
            print(f"[BIAS_RESET] Bias covariance reduced by factor of 10")
        
        # Reset stationary period tracking
        self._stationary_start_time = None
        
        return b_omega_reset, b_acc_reset, P_reset, True

    def detect_stationary_periods(self, timestamps, velocities, imu_data, 
                                velocity_threshold=0.1, min_duration=2.0, 
                                imu_consistency_check=True, verbose=False):
        """
        Detect periods when the vehicle is stationary based on velocity and IMU consistency.
        
        Args:
            timestamps (numpy.ndarray): Array of timestamps (N,)
            velocities (numpy.ndarray): Array of velocity vectors (N, 3)
            imu_data (numpy.ndarray): Array of IMU measurements (N, 6) [omega, accel]
            velocity_threshold (float): Velocity magnitude threshold for stationary detection
            min_duration (float): Minimum duration for a valid stationary period
            imu_consistency_check (bool): Whether to check IMU consistency during stationary periods
            verbose (bool): Enable verbose logging
            
        Returns:
            numpy.ndarray: Boolean array indicating stationary periods (N,)
        """
        
        N = len(timestamps)
        stationary_mask = np.zeros(N, dtype=bool)
        
        # Step 1: Find points where velocity is below threshold
        velocity_magnitudes = np.linalg.norm(velocities, axis=1)
        low_velocity_mask = velocity_magnitudes < velocity_threshold
        
        if verbose:
            print(f"[STATIONARY_DETECT] Found {np.sum(low_velocity_mask)} low velocity points out of {N}")
        
        # Step 2: Find continuous periods of low velocity
        # Find transitions
        transitions = np.diff(np.concatenate([[False], low_velocity_mask, [False]]).astype(int))
        starts = np.where(transitions == 1)[0]
        ends = np.where(transitions == -1)[0]
        
        for start_idx, end_idx in zip(starts, ends):
            # Check duration
            duration = timestamps[end_idx-1] - timestamps[start_idx]
            
            if duration < min_duration:
                if verbose:
                    print(f"[STATIONARY_DETECT] Period {start_idx}-{end_idx} too short: {duration:.2f}s")
                continue
            
            # IMU consistency check during this period
            if imu_consistency_check:
                period_imu = imu_data[start_idx:end_idx]
                
                # Check if gyroscope readings are consistent (low variation)
                omega_std = np.std(period_imu[:, :3], axis=0)
                max_omega_std = 0.02  # rad/s
                
                # Check if accelerometer readings are consistent 
                # (should be close to gravity, with low variation)
                acc_std = np.std(period_imu[:, 3:6], axis=0)
                max_acc_std = 0.2  # m/s²
                
                if np.any(omega_std > max_omega_std) or np.any(acc_std > max_acc_std):
                    if verbose:
                        print(f"[STATIONARY_DETECT] Period {start_idx}-{end_idx} failed IMU consistency:")
                        print(f"  Omega std: {omega_std} (max: {max_omega_std})")
                        print(f"  Acc std: {acc_std} (max: {max_acc_std})")
                    continue
            
            # Mark this period as stationary
            stationary_mask[start_idx:end_idx] = True
            
            if verbose:
                print(f"[STATIONARY_DETECT] Valid stationary period: {start_idx}-{end_idx} "
                    f"({duration:.2f}s, {end_idx-start_idx} samples)")
        
        if verbose:
            total_stationary_time = np.sum(stationary_mask) * np.mean(np.diff(timestamps))
            print(f"[STATIONARY_DETECT] Total stationary time: {total_stationary_time:.2f}s "
                f"({np.sum(stationary_mask)}/{N} samples)")
        
        return stationary_mask

    def run_with_bias_reset(self, t, u, measurements_covs, v_mes, p_mes0, N, ang0, 
                        enable_bias_reset=True, bias_reset_params=None, zupt_flags=None):
        """
        Run the IEKF with automatic IMU bias reset during stationary periods.
        
        Args:
            t (numpy.ndarray): Timestamps (seconds)
            u (numpy.ndarray): IMU measurements [angular_velocity (3), acceleration (3)]
            measurements_covs (numpy.ndarray): Measurement covariances for zero velocity constraints
            v_mes (numpy.ndarray): velocity measurements
            p_mes0 (numpy.ndarray): Initial position measurement
            N (int, optional): Number of time steps to process
            ang0 (numpy.ndarray): Initial orientation as [roll, pitch, yaw] in radians
            enable_bias_reset (bool): Whether to enable automatic bias reset
            bias_reset_params (dict): Parameters for bias reset (duration_threshold, velocity_threshold, etc.)
            zupt_flags (numpy.ndarray, optional): Precomputed ZUPT flags
            
        Returns:
            dict: Extended results dictionary including bias reset information
        """
        
        # Set default bias reset parameters
        if bias_reset_params is None:
            bias_reset_params = {
                'zupt_duration_threshold': 3.0,
                'velocity_threshold': 0.1,
                'bias_reset_factor': 0.0,
                'verbose': self.verbose
            }
        
        dt = t[1:] - t[:-1]  # (s)
        if N is None:
            N = u.shape[0]
        
        # Detect stationary periods if not provided
        if zupt_flags is None and enable_bias_reset:
            # We'll compute this online during the run
            zupt_flags = np.zeros(N, dtype=bool)
        elif zupt_flags is None:
            zupt_flags = np.zeros(N, dtype=bool)
        
        # Initialize arrays to store results
        Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P = self.init_run(p_mes0, v_mes[0], ang0, N)
        
        # Additional arrays to track bias resets
        bias_reset_events = []  # List of (timestamp, old_bias, new_bias) tuples
        
        for i in range(1, N):
            # Propagation step
            Rot[i], v[i], p[i], b_omega[i], b_acc[i], Rot_c_i[i], t_c_i[i], P = \
                self.propagate(Rot[i-1], v[i-1], p[i-1], b_omega[i-1], b_acc[i-1], Rot_c_i[i-1],
                            t_c_i[i-1], P, u[i], dt[i-1])
            
            # Update step with ZUPT if flagged
            Rot[i], v[i], p[i], b_omega[i], b_acc[i], Rot_c_i[i], t_c_i[i], P = \
                self.update(Rot[i], v[i], p[i], b_omega[i], b_acc[i], Rot_c_i[i], t_c_i[i], P, u[i],
                            i, measurements_covs[i], v_mes[i])
            
            # Check for bias reset if enabled
            if enable_bias_reset:
                # Store current time in internal tracker for bias reset function
                self._current_time = t[i]
                
                old_b_omega = b_omega[i].copy()
                old_b_acc = b_acc[i].copy()
                
                b_omega[i], b_acc[i], P, bias_was_reset = self.reset_imu_bias_with_zupt(
                    Rot[i], v[i], p[i], b_omega[i], b_acc[i], Rot_c_i[i], t_c_i[i], P, u[i],v_mes[i],
                    **bias_reset_params
                )
                
                # Record bias reset event
                if bias_was_reset:
                    bias_reset_events.append({
                        'timestamp': t[i],
                        'index': i,
                        'old_b_omega': old_b_omega,
                        'new_b_omega': b_omega[i],
                        'old_b_acc': old_b_acc, 
                        'new_b_acc': b_acc[i],
                        'omega_change': np.linalg.norm(b_omega[i] - old_b_omega),
                        'acc_change': np.linalg.norm(b_acc[i] - old_b_acc)
                    })
                    
                    if self.verbose:
                        print(f"[RUN] Bias reset at t={t[i]:.2f}s (step {i})")
            
            # Normalize rotations periodically
            if i % self.n_normalize_rot == 0:
                Rot[i] = self.normalize_rot(Rot[i])
            
            if i % self.n_normalize_rot_c_i == 0:
                Rot_c_i[i] = self.normalize_rot(Rot_c_i[i])

        return Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i


    def propagate(self, Rot_prev, v_prev, p_prev, b_omega_prev, b_acc_prev, Rot_c_i_prev,
                  t_c_i_prev, P_prev, u, dt):
        """
        Propagate the state forward using IMU measurements.

        This method implements the prediction step of the Kalman filter, integrating
        angular velocity and acceleration measurements from the IMU to predict the next state.

        Args:
            Rot_prev (numpy.ndarray): Previous orientation matrix (3x3)
            v_prev (numpy.ndarray): Previous velocity vector (3)
            p_prev (numpy.ndarray): Previous position vector (3)
            b_omega_prev (numpy.ndarray): Previous gyroscope bias (3)
            b_acc_prev (numpy.ndarray): Previous accelerometer bias (3)
            Rot_c_i_prev (numpy.ndarray): Previous car-to-IMU rotation matrix (3x3)
            t_c_i_prev (numpy.ndarray): Previous car-to-IMU translation vector (3)
            P_prev (numpy.ndarray): Previous state covariance matrix
            u (numpy.ndarray): Current IMU measurement [angular_velocity (3), acceleration (3)]
            dt (float): Time step in seconds

        Returns:
            tuple:
                - Rot (numpy.ndarray): Propagated orientation matrix (3x3)
                - v (numpy.ndarray): Propagated velocity vector (3)
                - p (numpy.ndarray): Propagated position vector (3)
                - b_omega (numpy.ndarray): Propagated gyroscope bias (3)
                - b_acc (numpy.ndarray): Propagated accelerometer bias (3)
                - Rot_c_i (numpy.ndarray): Propagated car-to-IMU rotation matrix (3x3)
                - t_c_i (numpy.ndarray): Propagated car-to-IMU translation vector (3)
                - P (numpy.ndarray): Propagated state covariance matrix
        """
        if self.verbose:
            print(f"[PROPAGATE] dt: {dt:.6f}s")
            print(f"[PROPAGATE] Input IMU - omega: {u[:3]}, acc: {u[3:6]}")
            print(f"[PROPAGATE] Previous state - v: {v_prev}, p: {p_prev}")
            print(f"[PROPAGATE] Previous biases - b_omega: {b_omega_prev}, b_acc: {b_acc_prev}")
        
        # Calculate acceleration in world frame, accounting for bias and gravity
        acc_corrected = u[3:6] - b_acc_prev
        acc = Rot_prev.dot(acc_corrected) + self.g
        
        if self.verbose:
            print(f"[PROPAGATE] Corrected acceleration (IMU frame): {acc_corrected}")
            print(f"[PROPAGATE] World frame acceleration (with gravity): {acc}")

        # Integrate acceleration to get velocity
        v = v_prev + acc * dt

        # Integrate velocity to get position (with acceleration contribution)
        p = p_prev + v_prev*dt + 1/2 * acc * dt**2

        # Calculate angular velocity with bias correction
        omega = u[:3] - b_omega_prev
        if self.verbose:
            print(f"[PROPAGATE] Corrected angular velocity: {omega}")

        # Integrate angular velocity to update orientation
        # so3exp converts the rotation vector to a rotation matrix
        omega_dt = omega * dt
        Rot = Rot_prev.dot(self.so3exp(omega_dt))

        if self.verbose:
            print(f"[PROPAGATE] Rotation increment: {omega_dt}")
            print(f"[PROPAGATE] New state - v: {v}, p: {p}")
            
            # Check for numerical issues
            rot_det = np.linalg.det(Rot)
            print(f"[PROPAGATE] Rotation matrix determinant: {rot_det:.6f} (should be ~1.0)")
            if abs(rot_det - 1.0) > 0.01:
                print(f"[PROPAGATE] WARNING: Rotation matrix determinant far from 1.0!")

        # Biases and car-to-IMU transform remain constant during propagation
        b_omega = b_omega_prev
        b_acc = b_acc_prev
        Rot_c_i = Rot_c_i_prev
        t_c_i = t_c_i_prev

        # Propagate the covariance matrix
        P = self.propagate_cov(P_prev, Rot_prev, v_prev, p_prev, b_omega_prev,
                               b_acc_prev, Rot_c_i, u, dt)
        
        if self.verbose:
            P_trace = np.trace(P)
            P_max_eigenval = np.max(np.linalg.eigvals(P))
            print(f"[PROPAGATE] Covariance trace: {P_trace:.6f}, max eigenvalue: {P_max_eigenval:.6f}")

        return Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P

    def propagate_cov(self, P_prev, Rot_prev, v_prev, p_prev, b_omega_prev,
                      b_acc_prev, Rot_c_i, u, dt):
        """
        Propagate the state covariance matrix.

        This method implements the covariance propagation for the Invariant Extended Kalman Filter.
        It builds the state transition matrix F and the process noise matrix G, then uses them
        to update the covariance matrix P.

        Args:
            P_prev (numpy.ndarray): Previous state covariance matrix
            Rot_prev (numpy.ndarray): Previous orientation matrix (3x3)
            v_prev (numpy.ndarray): Previous velocity vector (3)
            p_prev (numpy.ndarray): Previous position vector (3)
            b_omega_prev (numpy.ndarray): Previous gyroscope bias (3)
            b_acc_prev (numpy.ndarray): Previous accelerometer bias (3)
            u (numpy.ndarray): Current IMU measurement [angular_velocity (3), acceleration (3)]
            dt (float): Time step in seconds

        Returns:
            numpy.ndarray: Propagated state covariance matrix
        """
        # Initialize state transition and process noise matrices
        F = np.zeros((self.P_dim, self.P_dim))
        G = np.zeros((self.P_dim, self.Q_dim))

        # Compute skew-symmetric matrices for velocity and position
        v_skew_rot = self.skew(v_prev).dot(Rot_prev)
        p_skew_rot = self.skew(p_prev).dot(Rot_prev)

        # Build the state transition matrix F
        # Effect of orientation on velocity (through gravity)
        F[3:6, :3] = self.skew(self.g)
        # Effect of velocity on position
        F[6:9, 3:6] = self.Id3
        # Effect of gyroscope bias on orientation, velocity, and position
        F[:3, 9:12] = -Rot_prev
        F[3:6, 9:12] = -v_skew_rot
        F[6:9, 9:12] = -p_skew_rot
        # Effect of accelerometer bias on velocity
        F[3:6, 12:15] = -Rot_prev


        # Build the process noise matrix G
        # Effect of gyroscope noise on orientation, velocity, and position
        G[:3, :3] = Rot_prev
        G[3:6, :3] = v_skew_rot
        G[6:9, :3] = p_skew_rot
        # Effect of accelerometer noise on velocity
        G[3:6, 3:6] = Rot_prev
        G[9:15, 6:12] = self.Id6  # -self.Id6
        # Effect of car-to-IMU transform drift
        G[15:18, 12:15] = self.Id3 # -Rot_c_i
        G[18:21, 15:18] = self.Id3 #-self.Id3

        # Scale by dt and compute state transition matrix
        F = F * dt
        G = G * dt
        # Compute matrix exponential using Taylor series (3rd order approximation)
        F_square = F.dot(F)
        F_cube = F_square.dot(F)
        Phi = self.IdP + F + 1/2*F_square + 1/6*F_cube

        # Propagate covariance: P = Phi * (P + GQG') * Phi'
        P = Phi.dot(P_prev + G.dot(self.Q).dot(G.T)).dot(Phi.T)

        return P

    def update(self, Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P, u, i, measurement_cov, v_meas):
        """
        Update the state using zero lateral and vertical velocity constraints.

        This method implements the correction step of the Kalman filter, applying
        the zero lateral and vertical velocity constraints of ground vehicles.
        The constraint states that a vehicle on the ground has zero velocity
        in the lateral and vertical directions relative to the vehicle body frame.

        Args:
            Rot (numpy.ndarray): Current orientation matrix (3x3)
            v (numpy.ndarray): Current velocity vector (3)
            p (numpy.ndarray): Current position vector (3)
            b_omega (numpy.ndarray): Current gyroscope bias (3)
            b_acc (numpy.ndarray): Current accelerometer bias (3)
            Rot_c_i (numpy.ndarray): Current car-to-IMU rotation matrix (3x3)
            t_c_i (numpy.ndarray): Current car-to-IMU translation vector (3)
            P (numpy.ndarray): Current state covariance matrix
            u (numpy.ndarray): Current IMU measurement [angular_velocity (3), acceleration (3)]
            i (int): Current time step
            measurement_cov (numpy.ndarray): Measurement noise covariance for lateral and vertical velocity
            v_meas (numpy.ndarray): Current velocity measurement

        Returns:
            tuple:
                - Rot_up (numpy.ndarray): Updated orientation matrix (3x3)
                - v_up (numpy.ndarray): Updated velocity vector (3)
                - p_up (numpy.ndarray): Updated position vector (3)
                - b_omega_up (numpy.ndarray): Updated gyroscope bias (3)
                - b_acc_up (numpy.ndarray): Updated accelerometer bias (3)
                - Rot_c_i_up (numpy.ndarray): Updated car-to-IMU rotation matrix (3x3)
                - t_c_i_up (numpy.ndarray): Updated car-to-IMU translation vector (3)
                - P_up (numpy.ndarray): Updated state covariance matrix
        """
        # # Calculate orientation of body frame
        # Rot_body = Rot.dot(Rot_c_i)
        # # velocity in imu frame
        # v_imu = Rot.T.dot(v)
        # # velocity in body frame
        # v_body = Rot_c_i.T.dot(v_imu)
        # # velocity in body frame in the vehicle axis
        # v_body += self.skew(t_c_i).dot(u[:3] - b_omega)
        # Omega = self.skew(u[:3] - b_omega)

        # # Jacobian w.r.t. car frame
        # H_v_imu = Rot_c_i.T.dot(self.skew(v_imu))
        # H_t_c_i = -self.skew(t_c_i)

        # H = np.zeros((3, self.P_dim))
        # H[:, 3:6] = Rot_body.T
        # H[:, 15:18] = H_v_imu
        # H[:, 9:12] = H_t_c_i
        # H[:, 18:21] = -Omega
        # r = v_meas - v_body

        Omega = self.skew(u[:3] - b_omega)  # skew of angular velocity
        # orientation of body frame
        Rot_body = Rot.dot(Rot_c_i)
        # velocity in body frame in the imu axis
        v_imu = Rot.T.dot(v)
        v_body = Rot_c_i.T.dot(v_imu + Omega.dot(t_c_i))   # velocity in body frame in the vehicle axis
        # Jacobian w.r.t. car frame
        H_v_imu = self.skew(v_imu + Omega.dot(t_c_i)) 
        H_t_c_i = self.skew(t_c_i)
        H = np.zeros((3, self.P_dim))  # HH is a 3x21 matrix
        H[:, 3:6] = Rot_body.T  
        H[:, 9:12] = Rot_c_i.T.dot(H_t_c_i)
        H[:, 15:18] = Rot_c_i.T.dot(H_v_imu)  # Jacobian of delta_imu_car_rotation_extrinsic 
        H[:, 18:21] = Rot_c_i.T.dot(Omega)    # Jacobian of delta__imu_car_translation_extrinsic
        r = v_meas - v_body  # r is the residual between measurement, which is just difference between a 2x1 zero vector and v_body[1:],  
                           # v_body[1] is the lateral speed, v_body[2] is the upward speed


        if self.verbose:
            print(f"[UPDATE] Velocity in IMU frame: {v_imu}")
            print(f"[UPDATE] Velocity in body frame: {v_body}")
            print(f"[UPDATE] Angular velocity (corrected): {u[:3] - b_omega}")

        # Construct the measurement noise covariance matrix
        R = np.diag(measurement_cov)

        if self.verbose:
            print(f"[UPDATE] Measurement residual: {r}")
            print(f"[UPDATE] Residual norm: {np.linalg.norm(r):.6f}")
            
            # Check if measurement Jacobian is well-conditioned
            H_rank = np.linalg.matrix_rank(H)
            H_cond = np.linalg.cond(H.dot(H.T))
            print(f"[UPDATE] Measurement Jacobian rank: {H_rank}, condition number: {H_cond:.2e}")

            # Compute innovation covariance for debugging
            S = H.dot(P).dot(H.T) + R
            S_cond = np.linalg.cond(S)
            S_det = np.linalg.det(S)
            print(f"[UPDATE] Innovation covariance condition: {S_cond:.2e}, determinant: {S_det:.2e}")

        # Apply the Kalman filter update
        Rot_up, v_up, p_up, b_omega_up, b_acc_up, Rot_c_i_up, t_c_i_up, P_up = \
            self.state_and_cov_update(Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P, H, r, R)

        if  self.verbose:
            # Check state changes
            v_change = np.linalg.norm(v_up - v)
            p_change = np.linalg.norm(p_up - p)
            bias_omega_change = np.linalg.norm(b_omega_up - b_omega)
            bias_acc_change = np.linalg.norm(b_acc_up - b_acc)
            
            print(f"[UPDATE] State changes:")
            print(f"  - Velocity change: {v_change:.6f}")
            print(f"  - Position change: {p_change:.6f}")
            print(f"  - Gyro bias change: {bias_omega_change:.6f}")
            print(f"  - Acc bias change: {bias_acc_change:.6f}")
            
            # Check covariance update
            P_trace_before = np.trace(P)
            P_trace_after = np.trace(P_up)
            print(f"[UPDATE] Covariance trace change: {P_trace_before:.6f} -> {P_trace_after:.6f}")
            
            # Check if covariance decreased (as expected after measurement update)
            if P_trace_after > P_trace_before:
                print(f"[UPDATE] WARNING: Covariance trace increased after update!")

        return Rot_up, v_up, p_up, b_omega_up, b_acc_up, Rot_c_i_up, t_c_i_up, P_up

    @staticmethod
    def state_and_cov_update(Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P, H, r, R):
        """
        Update the state and covariance matrix based on the measurement residual.

        This static method implements the Kalman filter update equations, specifically
        adapted for use with Lie groups for the rotation matrices. The state update
        is done on the error state in the Lie algebra, then mapped back to the state space.

        Args:
            Rot (numpy.ndarray): Current orientation matrix (3x3)
            v (numpy.ndarray): Current velocity vector (3)
            p (numpy.ndarray): Current position vector (3)
            b_omega (numpy.ndarray): Current gyroscope bias (3)
            b_acc (numpy.ndarray): Current accelerometer bias (3)
            Rot_c_i (numpy.ndarray): Current car-to-IMU rotation matrix (3x3)
            t_c_i (numpy.ndarray): Current car-to-IMU translation vector (3)
            P (numpy.ndarray): Current state covariance matrix
            H (numpy.ndarray): Measurement Jacobian matrix
            r (numpy.ndarray): Measurement residual vector
            R (numpy.ndarray): Measurement noise covariance matrix

        Returns:
            tuple:
                - Rot_up (numpy.ndarray): Updated orientation matrix (3x3)
                - v_up (numpy.ndarray): Updated velocity vector (3)
                - p_up (numpy.ndarray): Updated position vector (3)
                - b_omega_up (numpy.ndarray): Updated gyroscope bias (3)
                - b_acc_up (numpy.ndarray): Updated accelerometer bias (3)
                - Rot_c_i_up (numpy.ndarray): Updated car-to-IMU rotation matrix (3x3)
                - t_c_i_up (numpy.ndarray): Updated car-to-IMU translation vector (3)
                - P_up (numpy.ndarray): Updated state covariance matrix
        """
        # Compute innovation covariance
        S = H.dot(P).dot(H.T) + R

        # Compute Kalman gain
        # Equivalent to K = P.dot(H.T).dot(np.linalg.inv(S)) but more numerically stable
        K = (np.linalg.solve(S, P.dot(H.T).T)).T

        # Compute state correction vector
        dx = K.dot(r)

        # Apply correction to SE(3) state components (orientation, velocity, position)
        # sen3exp maps the Lie algebra correction to a transformation on SE(3)
        dR, dxi = NUMPYIEKF.sen3exp(dx[:9])
        dv = dxi[:, 0]  # Velocity correction
        dp = dxi[:, 1]  # Position correction

        # Apply corrections to state
        Rot_up = dR.dot(Rot)  # Right-multiply for orientation update
        v_up = dR.dot(v) + dv  # Apply rotation and translation to velocity
        p_up = dR.dot(p) + dp  # Apply rotation and translation to position

        # Apply corrections to bias states (simple addition)
        b_omega_up = b_omega + dx[9:12] * 0.1
        b_acc_up = b_acc + dx[12:15] * 0.1

        # Apply correction to car-to-IMU rotation
        dR = NUMPYIEKF.so3exp(dx[15:18])
        Rot_c_i_up = dR.dot(Rot_c_i)

        # Apply correction to car-to-IMU translation
        t_c_i_up = t_c_i + dx[18:21]

        # Update covariance with Joseph form for numerical stability
        I_KH = NUMPYIEKF.IdP - K.dot(H)
        P_up = I_KH.dot(P).dot(I_KH.T) + K.dot(R).dot(K.T)

        # Enforce symmetry of covariance matrix
        P_up = (P_up + P_up.T)/2

        return Rot_up, v_up, p_up, b_omega_up, b_acc_up, Rot_c_i_up, t_c_i_up, P_up

    @staticmethod
    def skew(x):
        """
        Create a skew-symmetric matrix from a 3D vector.

        The skew-symmetric matrix is useful for representing cross products as matrix
        multiplications and is used extensively in Lie group operations for rotations.

        Args:
            x (numpy.ndarray): 3D vector [x, y, z]

        Returns:
            numpy.ndarray: 3x3 skew-symmetric matrix of the form:
                [  0  -z   y ]
                [  z   0  -x ]
                [ -y   x   0 ]
        """
        X = np.array([[0, -x[2], x[1]],
                      [x[2], 0, -x[0]],
                      [-x[1], x[0], 0]])
        return X

    @staticmethod
    def rot_from_2_vectors(v1, v2):
        """
        Compute the rotation matrix that rotates vector v1 to align with vector v2.

        This uses the Rodrigues formula to find the rotation matrix that maps
        one vector to another. The vectors are first normalized to unit length.

        Args:
            v1 (numpy.ndarray): First 3D vector
            v2 (numpy.ndarray): Second 3D vector

        Returns:
            numpy.ndarray: 3x3 rotation matrix that rotates v1 to align with v2
        """
        # Normalize input vectors to unit length
        v1 = v1/np.linalg.norm(v1)
        v2 = v2/np.linalg.norm(v2)

        # Compute cross product between the vectors
        v = np.cross(v1, v2)

        # Compute cosine and sine of the angle between the vectors
        cosang = np.dot(v1, v2)
        sinang = np.linalg.norm(v)

        # Compute rotation matrix using Rodrigues formula
        # R = I + sin(θ)[v]_× + (1-cos(θ))[v]_×²
        Rot = NUMPYIEKF.Id3 + NUMPYIEKF.skew(v) + \
              NUMPYIEKF.skew(v).dot(NUMPYIEKF.skew(v))*(1-cosang)/(sinang**2)

        # Ensure the result is a valid rotation matrix
        Rot = NUMPYIEKF.normalize_rot(Rot)

        return Rot

    @staticmethod
    def sen3exp(xi):
        """
        Exponential map for the Lie algebra sen(3) to the Lie group SEN(3).

        This function maps an element of the Lie algebra sen(3) to the corresponding
        element of the Lie group SEN(3), which represents rigid body transformations.
        It is used in the IEKF update step to apply corrections to the state.

        The input vector xi = [phi, rho] represents a screw motion, where:
        - phi (3D): Rotation vector (axis-angle representation)
        - rho (3Nx3): Translation components for N points

        Args:
            xi (numpy.ndarray): sen(3) Lie algebra element [phi, rho]
                phi: 3D rotation vector
                rho: 3Nx3 translation components (reshaped to N points, each with 3D coords)

        Returns:
            tuple:
                - Rot (numpy.ndarray): 3x3 rotation matrix
                - x (numpy.ndarray): Nx3 transformed points
        """
        # Extract rotation component
        phi = xi[:3]

        # Compute the angle (norm of rotation vector)
        angle = np.linalg.norm(phi)

        # Near |phi|==0, use first order Taylor expansion to avoid numerical issues
        if np.abs(angle) < 1e-8:
            # Create skew-symmetric matrix directly
            skew_phi = np.array([[0, -phi[2], phi[1]],
                        [phi[2], 0, -phi[0]],
                        [-phi[1], phi[0], 0]])
            # Jacobian approximation for small angles
            J = NUMPYIEKF.Id3 + 0.5 * skew_phi
            # Rotation matrix approximation for small angles
            Rot = NUMPYIEKF.Id3 + skew_phi
        else:
            # Normalize rotation axis
            axis = phi / angle
            # Create skew-symmetric matrix of the axis
            skew_axis = np.array([[0, -axis[2], axis[1]],
                        [axis[2], 0, -axis[0]],
                        [-axis[1], axis[0], 0]])

            s = np.sin(angle)
            c = np.cos(angle)

            # Compute the left Jacobian of SO(3)
            # J = sin(θ)/θ·I + (1-sin(θ)/θ)·a·aᵀ + (1-cos(θ))/θ·[a]×
            J = (s / angle) * NUMPYIEKF.Id3 \
                   + (1 - s / angle) * np.outer(axis, axis) + ((1 - c) / angle) * skew_axis

            # Compute rotation matrix using Rodrigues' formula
            # R = cos(θ)·I + (1-cos(θ))·a·aᵀ + sin(θ)·[a]×
            Rot = c * NUMPYIEKF.Id3 + (1 - c) * np.outer(axis, axis) + s * skew_axis

        # Transform the translation components using the Jacobian
        # Reshape to handle multiple points
        x = J.dot(xi[3:].reshape(-1, 3).T)

        return Rot, x

    @staticmethod
    def so3exp(phi):
        """
        Exponential map for the Lie algebra so(3) to the Lie group SO(3).

        This function maps a 3D vector in the Lie algebra so(3) to a rotation matrix
        in the Lie group SO(3) using the Rodrigues formula. It is used to convert
        angular velocities or rotation vectors to rotation matrices.

        Args:
            phi (numpy.ndarray): 3D rotation vector (axis-angle representation)

        Returns:
            numpy.ndarray: 3x3 rotation matrix
        """
        # Compute the angle (norm of rotation vector)
        angle = np.linalg.norm(phi)

        # Near phi==0, use first order Taylor expansion to avoid numerical issues
        if np.abs(angle) < 1e-8:
            # Create skew-symmetric matrix directly
            skew_phi = np.array([[0, -phi[2], phi[1]],
                      [phi[2], 0, -phi[0]],
                      [-phi[1], phi[0], 0]])
            # For small angles, R ≈ I + [phi]×
            return np.identity(3) + skew_phi

        # Normalize rotation axis
        axis = phi / angle

        # Create skew-symmetric matrix of the axis
        skew_axis = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])

        s = np.sin(angle)
        c = np.cos(angle)

        # Compute rotation matrix using Rodrigues' formula
        # R = cos(θ)·I + (1-cos(θ))·a·aᵀ + sin(θ)·[a]×
        return c * NUMPYIEKF.Id3 + (1 - c) * np.outer(axis, axis) + s * skew_axis

    @staticmethod
    def so3left_jacobian(phi):
        """
        Compute the left Jacobian of SO(3).

        The left Jacobian of SO(3) is used for mapping incremental rotations and
        is essential for uncertainty propagation in the Lie algebra.

        Args:
            phi (numpy.ndarray): 3D rotation vector (axis-angle representation)

        Returns:
            numpy.ndarray: 3x3 left Jacobian matrix
        """
        # Compute the angle (norm of rotation vector)
        angle = np.linalg.norm(phi)

        # Near |phi|==0, use first order Taylor expansion to avoid numerical issues
        if np.abs(angle) < 1e-8:
            # Create skew-symmetric matrix directly
            skew_phi = np.array([[0, -phi[2], phi[1]],
                      [phi[2], 0, -phi[0]],
                      [-phi[1], phi[0], 0]])
            # For small angles, J ≈ I + 0.5[φ]×
            return NUMPYIEKF.Id3 + 0.5 * skew_phi

        # Normalize rotation axis
        axis = phi / angle

        # Create skew-symmetric matrix of the axis
        skew_axis = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])

        s = np.sin(angle)
        c = np.cos(angle)

        # Compute the left Jacobian using the formula:
        # J = sin(θ)/θ·I + (1-sin(θ)/θ)·a·aᵀ + (1-cos(θ))/θ·[a]×
        return (s / angle) * NUMPYIEKF.Id3 \
               + (1 - s / angle) * np.outer(axis, axis) + ((1 - c) / angle) * skew_axis

    @staticmethod
    def normalize_rot(Rot):
        """
        Normalize a rotation matrix to ensure it is a valid element of SO(3).

        This function projects a general 3x3 matrix to the SO(3) manifold using SVD.
        It is used to correct numerical errors that accumulate during integration.

        Args:
            Rot (numpy.ndarray): 3x3 matrix to be normalized

        Returns:
            numpy.ndarray: Valid 3x3 rotation matrix (orthogonal with determinant 1)
        """
        # Perform singular value decomposition
        # The SVD is commonly written as Rot = U·S·V.H.
        # The v returned by numpy's function is V.H and u = U.
        U, _, V = np.linalg.svd(Rot, full_matrices=False)

        # Ensure the resulting matrix has determinant 1 (proper rotation)
        S = np.eye(3)
        S[2, 2] = np.linalg.det(U) * np.linalg.det(V)

        # Reconstruct the rotation matrix
        return U.dot(S).dot(V)

    @staticmethod
    def from_rpy(roll, pitch, yaw):
        """
        Convert roll, pitch, yaw Euler angles to rotation matrix.

        This function creates a rotation matrix from Euler angles using the
        ZYX convention (yaw-pitch-roll, intrinsic rotations).

        Args:
            roll (float): Roll angle in radians (rotation around X-axis)
            pitch (float): Pitch angle in radians (rotation around Y-axis)
            yaw (float): Yaw angle in radians (rotation around Z-axis)

        Returns:
            numpy.ndarray: 3x3 rotation matrix
        """
        # Apply rotations in order: first roll (X), then pitch (Y), then yaw (Z)
        return NUMPYIEKF.rotz(yaw).dot(NUMPYIEKF.roty(pitch).dot(NUMPYIEKF.rotx(roll)))

    @staticmethod
    def rotx(t):
        """
        Create a rotation matrix for rotation around the X axis.

        Args:
            t (float): Angle in radians

        Returns:
            numpy.ndarray: 3x3 rotation matrix for rotation around X axis
        """
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[1,  0,  0],
                         [0,  c, -s],
                         [0,  s,  c]])

    @staticmethod
    def roty(t):
        """
        Create a rotation matrix for rotation around the Y axis.

        Args:
            t (float): Angle in radians

        Returns:
            numpy.ndarray: 3x3 rotation matrix for rotation around Y axis
        """
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c,  0,  s],
                         [0,  1,  0],
                         [-s, 0,  c]])

    @staticmethod
    def rotz(t):
        """
        Create a rotation matrix for rotation around the Z axis.

        Args:
            t (float): Angle in radians

        Returns:
            numpy.ndarray: 3x3 rotation matrix for rotation around Z axis
        """
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, -s,  0],
                         [s,  c,  0],
                         [0,  0,  1]])

    @staticmethod
    def to_rpy(Rot):
        """
        Convert a rotation matrix to roll, pitch, yaw Euler angles.

        This function extracts Euler angles from a rotation matrix using the
        ZYX convention (yaw-pitch-roll). Special care is taken to handle
        gimbal lock cases when pitch is ±90 degrees.

        Args:
            Rot (numpy.ndarray): 3x3 rotation matrix

        Returns:
            tuple: (roll, pitch, yaw) angles in radians
        """
        # Extract pitch angle using atan2 for correct quadrant
        pitch = np.arctan2(-Rot[2, 0], np.sqrt(Rot[0, 0]**2 + Rot[1, 0]**2))

        # Handle gimbal lock cases (pitch = ±90°)
        if np.isclose(pitch, np.pi / 2.):
            # When pitch = 90°, yaw and roll are linked (gimbal lock)
            # We choose to set yaw = 0 and compute roll
            yaw = 0.
            roll = np.arctan2(Rot[0, 1], Rot[1, 1])
        elif np.isclose(pitch, -np.pi / 2.):
            # When pitch = -90°, yaw and roll are linked (gimbal lock)
            # We choose to set yaw = 0 and compute roll
            yaw = 0.
            roll = -np.arctan2(Rot[0, 1], Rot[1, 1])
        else:
            # Normal case - no gimbal lock
            # Compute secant of pitch for efficiency
            sec_pitch = 1. / np.cos(pitch)
            # Extract yaw and roll using atan2 for correct quadrant
            yaw = np.arctan2(Rot[1, 0] * sec_pitch,
                             Rot[0, 0] * sec_pitch)
            roll = np.arctan2(Rot[2, 1] * sec_pitch,
                              Rot[2, 2] * sec_pitch)

        return roll, pitch, yaw

    def set_learned_covariance(self, torch_iekf):
        """
        Set the filter's covariance parameters from a trained PyTorch IEKF model.

        This method is used to transfer learned covariance parameters from a PyTorch
        implementation of the IEKF (which can be trained using deep learning) to this
        NumPy implementation (which is used for evaluation).

        Args:
            torch_iekf: A PyTorch IEKF model with trained covariance parameters

        Note:
            This method performs the following transfers:
            1. Process noise covariance matrix Q from the PyTorch model
            2. Initial state covariance scaling factors from the model's initprocesscov_net
        """
        # Update the process noise covariance matrix Q
        torch_iekf.set_Q()  # Ensure the PyTorch model has updated its Q matrix
        self.Q = torch_iekf.Q.cpu().detach().numpy()  # Transfer to NumPy

        # Get the scaling factors for initial covariances from the network
        beta = torch_iekf.initprocesscov_net.init_cov(torch_iekf)\
            .detach().cpu().numpy()

        # Apply scaling factors to the initial covariances
        self.cov_Rot0 *= beta[0]           # Initial orientation covariance
        self.cov_v0 *= beta[1]             # Initial velocity covariance
        self.cov_b_omega0 *= beta[2]       # Initial gyro bias covariance
        self.cov_b_acc0 *= beta[3]         # Initial accelerometer bias covariance
        self.cov_Rot_c_i0 *= beta[4]       # Initial car-to-IMU rotation covariance
        self.cov_t_c_i0 *= beta[5]         # Initial car-to-IMU translation covariance
