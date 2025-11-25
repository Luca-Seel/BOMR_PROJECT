import numpy as np

# EKF
Ts = 0.1  # time step in seconds
L = 95  # distance between wheels in mm
speed_to_mms = 0.3375  # conversion factor from thymio speed units to mm/s from solution ex.8 (in our measurement it was 0.43478260869565216)

def wrap_angle(theta):
    return (theta + np.pi) % (2*np.pi) - np.pi

def motors_to_vw(vl, vr, speed_to_mms, L):
    v_l = vl * speed_to_mms
    v_r = vr * speed_to_mms
    v = (v_l + v_r) / 2.0
    w = (v_r - v_l) / L
    return v, w

def joseph_update(P, K, H, R): # better numerical robustness than  P_upd = (np.eye(5) - K @ H) @ P_pred
    I = np.eye(P.shape[0])
    IKH = I - K @ H
    return IKH @ P @ IKH.T + K @ R @ K.T

def ekf_predict(x_prev, P_prev, u, Ts, q_proc):
    x, y, th, v, om = x_prev
    v_ctrl, om_ctrl = u

    x_pred = np.array([
        x + v_ctrl * Ts * np.cos(th),
        y + v_ctrl * Ts * np.sin(th),
        wrap_angle(th + om_ctrl * Ts),
        v_ctrl,
        om_ctrl
    ])

    F = np.array([
        [1, 0, -v_ctrl * Ts * np.sin(th),  Ts * np.cos(th), 0],
        [0, 1,  v_ctrl * Ts * np.cos(th),  Ts * np.sin(th), 0],
        [0, 0, 1,                          0,               Ts],
        [0, 0, 0,                          1,               0 ],
        [0, 0, 0,                          0,               1 ],
    ])

    Gc = np.array([
        [Ts * np.cos(th),  0],
        [Ts * np.sin(th),  0],
        [0,                Ts],
        [1,                0],
        [0,                1]
    ])

    q_x, q_y, q_th, q_v_ctrl, q_om_ctrl, q_v_bias, q_om_bias = q_proc
    Q_ctrl = np.diag([q_v_ctrl, q_om_ctrl])
    Q_bias = np.diag([q_x, q_y, q_th, q_v_bias, q_om_bias])
    Q = Gc @ Q_ctrl @ Gc.T + Q_bias

    P_pred = F @ P_prev @ F.T + Q
    return x_pred, P_pred


def ekf_update_cam(x_pred, P_pred, z_cam, r_cam):
    H = np.array([
        [1,0,0,0,0],
        [0,1,0,0,0],
        [0,0,1,0,0],
    ])
    R = np.diag(r_cam)
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    y_residual = z_cam - H @ x_pred
    y_residual[2] = wrap_angle(y_residual[2])
    x_upd = x_pred + K @ y_residual
    x_upd[2] = wrap_angle(x_upd[2])
    P_upd = (np.eye(5) - K @ H) @ P_pred
    #P_upd = joseph_update(P_pred, K, H, R)
    return x_upd, P_upd

def ekf_update_both(x_pred, P_pred, z_cam, r_cam, z_mot, r_mot):
    H = np.eye(5)
    R = np.block([
        [np.diag(r_cam),  np.zeros((3,2))],
        [np.zeros((2,3)), np.diag(r_mot)]
        ])
    z = np.concatenate([z_cam, z_mot])
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    y_residual = z - H @ x_pred
    y_residual[2] = wrap_angle(y_residual[2])
    x_upd = x_pred + K @ y_residual
    x_upd[2] = wrap_angle(x_upd[2])
    P_upd = (np.eye(5) - K @ H) @ P_pred
    #P_upd = joseph_update(P_pred, K, H, R)
    return x_upd, P_upd

def ekf_update_motors(x_pred, P_pred, z_mot, r_mot):
    H = np.array([
        [0,0,0,1,0],
        [0,0,0,0,1],
    ])
    R = np.diag(r_mot)
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    y_residual = z_mot - H @ x_pred
    x_upd = x_pred + K @ y_residual
    x_upd[2] = wrap_angle(x_upd[2])
    P_upd = (np.eye(5) - K @ H) @ P_pred
    #P_upd = joseph_update(P_pred, K, H, R)
    return x_upd, P_upd
