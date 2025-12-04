import numpy as np
import matplotlib.pyplot as plt
# EKF
Ts = 0.1  # time step in seconds
L = 95  # distance between wheels in mm
speed_to_mms = 0.4347  # conversion factor from thymio speed units to mm/s

# Process noise for EKF (from model-mismatch/random-walk/control execution)
q_proc = (
    1e-3, 1e-2, 1e-4,   # q_x, q_y, q_theta (model mismatch)
    75.72,  0.002692,         # q_v_ctrl, q_omega_ctrl (control execution noise)
    1e-2, 1e-3          # q_v_bias, q_omega_bias (random walk on v, omega)
)
# Camera measurement noise
r_cam = (1.435, 1.864, 0.001496)  # [mm^2, mm^2, rad^2]
r_mot = (75.72, 0.002692)    # motor noise on v, omega


class EKFState:
    def __init__(self, x0=None, P0=None):
        self.x = np.zeros(5) if x0 is None else np.array(x0, dtype=float)
        self.P = 1000.0*np.eye(5) if P0 is None else np.array(P0, dtype=float)
        self.P_pred_prior = 1000.0*np.eye(5) if P0 is None else np.array(P0, dtype=float)

    def step(self, vl_cmd, vr_cmd, z_cam=None, r_cam=None, z_mot=None, r_mot=None, Ts=0.1, q_proc=None):
        # predict
        x_pred, P_pred = ekf_predict(self.x, self.P, vl_cmd, vr_cmd, Ts, q_proc)
        self.P_pred_prior = P_pred
        # update(s)
        if z_cam is not None and r_cam is not None:
            x_pred, P_pred = ekf_update_cam(x_pred, P_pred, z_cam, r_cam)
        if z_mot is not None and r_mot is not None:
            x_pred, P_pred = ekf_update_motors(x_pred, P_pred, z_mot, r_mot)
        self.x, self.P = x_pred, P_pred

    def get_state(self):
        # (x_mm, y_mm, theta_rad)
        return float(self.x[0]), float(self.x[1]), float(wrap_angle(self.x[2]))

def wrap_angle(theta):
    return (theta + np.pi) % (2*np.pi) - np.pi

def motors_to_vw(vl, vr, speed_to_mms, L):
    v_l = vl * speed_to_mms
    v_r = vr * speed_to_mms
    v = (v_l + v_r) / 2.0
    w = (v_l - v_r) / L
    return v, w

def joseph_update(P, K, H, R): # better numerical robustness than  P_upd = (np.eye(5) - K @ H) @ P_pred
    I = np.eye(P.shape[0])
    IKH = I - K @ H
    return IKH @ P @ IKH.T + K @ R @ K.T

def ekf_predict(x_prev, P_prev, vl_cmd, vr_cmd, Ts, q_proc):
    x, y, th, v, om = x_prev
    v_ctrl, om_ctrl = motors_to_vw(vl_cmd, vr_cmd, speed_to_mms, L)
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
    #P_upd = (np.eye(5) - K @ H) @ P_pred
    P_upd = joseph_update(P_pred, K, H, R)
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
    #P_upd = (np.eye(5) - K @ H) @ P_pred
    P_upd = joseph_update(P_pred, K, H, R)
    return x_upd, P_upd

