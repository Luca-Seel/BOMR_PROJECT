import numpy as np
import matplotlib.pyplot as plt
# EKF
Ts = 0.1  # time step in seconds
L = 95  # distance between wheels in mm
speed_to_mms = 0.3375  # conversion factor from thymio speed units to mm/s from solution ex.8 (in our measurement it was 0.43478260869565216)

# Process noise for EKF (tune) (from model-mismatch/random-walk/control execution)
q_proc = (
    1e-3, 1e-2, 1e-4,   # q_x, q_y, q_theta (model mismatch)
    75.72,  0.002692,         # q_v_ctrl, q_omega_ctrl (control execution noise)
    1e-2, 1e-5          # q_v_bias, q_omega_bias (random walk on v, omega)
)
# Camera measurement noise (tune)
r_cam = (1.435, 1.864, 0.001496)  # [mm^2, mm^2, rad^2]
r_mot = (75.72, 0.002692)    # motor noise on v, omega


class EKFState:
    def __init__(self, x0=None, P0=None):
        self.x = np.zeros(5) if x0 is None else np.array(x0, dtype=float)
        self.P = 1000.0*np.eye(5) if P0 is None else np.array(P0, dtype=float)

    def step(self, vl_cmd, vr_cmd, z_cam=None, r_cam=None, z_mot=None, r_mot=None, Ts=0.1, q_proc=None):
        # predict
        x_pred, P_pred = ekf_predict(self.x, self.P, vl_cmd, vr_cmd, Ts, q_proc)
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
    w = (v_r - v_l) / L
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

def init_Ppred_view(N_states=5, heatmap_update_every=20, log_scale=True, warmup_for_clim=10):
    plt.ion()

    # Figure and axes
    fig, (ax_diag, ax_hm) = plt.subplots(2, 1, figsize=(7, 7), constrained_layout=True)

    # Top: diag lines
    lines = [ax_diag.plot([], [], label=f'var[{i}]')[0] for i in range(N_states)]
    ax_diag.set_xlabel('time step')
    ax_diag.set_ylabel('variance (log10)' if log_scale else 'variance')
    ax_diag.grid(True)
    ax_diag.legend()

    # Bottom: heatmap
    im = ax_hm.imshow(np.eye(N_states), cmap='viridis', interpolation='nearest')
    cbar = plt.colorbar(im, ax=ax_hm)
    ax_hm.set_title('P_pred')
    ax_hm.set_xlabel('state j')
    ax_hm.set_ylabel('state i')

    # Internal state
    var_hist = [[] for _ in range(N_states)]
    k_hist = []
    clim_samples = []
    autoscale_diag = True
    state = {'step': -1, 'clim_fixed': False}

    def update(P_pred):
        # advance internal step counter
        state['step'] += 1
        s = state['step']

        # Top panel: variances
        diag = np.diag(P_pred).astype(float)
        vals = np.log10(np.maximum(diag, 1e-12)) if log_scale else diag

        k_hist.append(s)
        for i in range(N_states):
            var_hist[i].append(vals[i])
            lines[i].set_data(k_hist, var_hist[i])

        if autoscale_diag:
            ax_diag.relim()
            ax_diag.autoscale_view()

        # Heatmap color limit warmup/fix
        if not state['clim_fixed']:
            clim_samples.append(P_pred.copy())
            if s >= warmup_for_clim - 1:
                P_stack = np.stack(clim_samples, axis=0)
                vmin, vmax = float(np.min(P_stack)), float(np.max(P_stack))
                if vmax <= vmin:  # fallback
                    vmin, vmax = float(np.min(P_pred)), float(np.max(P_pred))
                im.set_clim(vmin=vmin, vmax=vmax)
                cbar.update_normal(im)
                state['clim_fixed'] = True

        # Update heatmap every N steps (always during warmup)
        if (s % heatmap_update_every == 0) or (not state['clim_fixed']):
            im.set_data(P_pred)

        plt.pause(0.001)

    return update