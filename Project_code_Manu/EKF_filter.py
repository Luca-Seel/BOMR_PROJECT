import numpy as np

# EKF
Ts = 0.1  # time step in seconds
L = 95  # distance between wheels in mm
speed_to_mms = 0.3375  # conversion factor from thymio speed units to mm/s from solution ex.8 (in our measurement it was 0.43478260869565216)

def ekf_filter(speed, pos, x_est_prev, P_est_prev, var_motors, var_camera):
    """
    Extended Kalman Filter for robot state estimation.  
    
    Parameters:
    speed: tuple  (v_l, v_r) - left and right wheel speeds in thymio speed units
    pos: tuple (x, y, theta) - position measurement from camera
    x_est_prev: np.array - previous state estimate [x, y, theta, v, omega]
    P_est_prev: np.array - previous covariance estimate (5x5 matrix)
    var_motors: tuple (var_speed, var_rot_speed) - variances of the motor measurement noise
    var_camera: tuple (var_x, var_y, var_theta) - variances of the camera measurement noise
    
    Returns:
    x_est: np.array - updated state estimate [x, y, theta, v, omega]
    P_est: np.array - updated covariance estimate (5x5 matrix)
    """
    
    
    # Prediction step:
    # State prediction
    theta_state = x_est_prev[2]
    v_state = x_est_prev[3]
    omega_state = x_est_prev[4]
    x_pred = np.array([
        x_est_prev[0] + v_state * Ts * np.cos(theta_state),
        x_est_prev[1] + v_state * Ts * np.sin(theta_state),
        theta_state + omega_state * Ts,
        v_state,
        omega_state
    ])

    # Jacobian of the motion model
    F = np.array([
        [1, 0, -v_state * Ts * np.sin(theta_state),  Ts * np.cos(theta_state), 0],
        [0, 1, v_state * Ts * np.cos(theta_state),  Ts * np.sin(theta_state), 0],
        [0, 0, 1, 0, Ts],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1]
    ])

    # Process noise covariance (motor noise)
    G = np.array([
        [Ts * np.cos(theta_state), 0],
        [Ts * np.sin(theta_state), 0],
        [0, Ts],
        [1, 0],
        [0, 1] 
    ])
    Q = np.dot(G, np.diag([var_motors[0], var_motors[1]]).dot(G.T)) # (5x5 matrix)

    # Covariance prediction
    P_pred = F @ P_est_prev @ F.T + Q # (5x5 matrix)

    # Measurement update step (from vision and motors)
    H = np.eye(5)  # measurement matrix (5x5 matrix)
    R = np.diag([var_camera[0], var_camera[1], var_camera[2], var_motors[0], var_motors[1]])  # measurement noise covariance (5x5 matrix)

    # Kalman gain
    S = H @ P_pred @ H.T + R # (5x5 matrix)
    K = P_pred @ H.T @ np.linalg.inv(S) # (5x5 matrix)

    # Measurement residual
    v_l, v_r = speed
    v_l = v_l * speed_to_mms  # convert to mm/s
    v_r = v_r * speed_to_mms  # convert to mm/s
    v = (v_l + v_r) / 2.0   # linear velocity
    omega = (v_r - v_l) / L  # angular velocity
    z = np.array([pos[0], pos[1], pos[2], v, omega]) # measured state
    y_residual = z - H @ x_pred

    # State update
    x_est = x_pred + K @ y_residual

    # Covariance update
    P_est = (np.eye(5) - K @ H) @ P_pred # (5x5 matrix)

    return x_est, P_est