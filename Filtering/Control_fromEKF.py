import math
from time import monotonic as now
import numpy as np

class CmdHolder:
    def __init__(self):
        self.vl = 0
        self.vr = 0

    def set(self, vl, vr):
        self.vl, self.vr = int(vl), int(vr)

    def get(self):
        return self.vl, self.vr

cmd_holder = CmdHolder()

def get_cmd():
    # used by the EKF task to retrieve last commanded wheel targets
    return cmd_holder.get()

def wrap_angle(theta):
    return (theta + math.pi) % (2*math.pi) - math.pi

# -------------------- Thymio I/O helpers --------------------

async def set_motors(node, left, right):
    # record last command for EKF
    cmd_holder.set(left, right)
    # send to Thymio
    await node.set_variables({
        "motor.left.target":  [int(left)],
        "motor.right.target": [int(right)],
    })

async def stop(node):
    await set_motors(node, 0, 0)

# -------------------- Heading control (EKF-based) --------------------

async def turn_to_heading(node, client, ekf_get_state, theta_ref,
                          w_cmd=150, dt=0.02, ang_tol=0.02, timeout=6.0):
    """Closed-loop in-place turn using EKF heading feedback."""
    t0 = now()
    while True:
        x, y, theta = ekf_get_state()[:3]
        e = wrap_angle(theta_ref - theta)
        if abs(e) <= ang_tol:
            break
        sgn = 1 if e >= 0 else -1
        await set_motors(node, -sgn*w_cmd, sgn*w_cmd)
        await client.sleep(dt)
        if now() - t0 > timeout:
            print("turn_to_heading: timeout")
            break
    await stop(node)

async def turn_toward_point(node, client, ekf_get_state, tx, ty,
                            w_cmd=150, dt=0.02, ang_tol=0.02, timeout=6.0):
    """Aim toward a target point (tx, ty) using EKF pose."""
    x, y, theta = ekf_get_state()[:3]
    dx, dy = tx - x, ty - y
    theta_ref = math.atan2(-dx, dy)  # +y forward, +x left (your convention)
    await turn_to_heading(node, client, ekf_get_state, theta_ref,
                          w_cmd=w_cmd, dt=dt, ang_tol=ang_tol, timeout=timeout)

# -------------------- Go-to-waypoint primitives (EKF) --------------------

async def move_to_position_ekf(node, client, ekf_get_state,
                               target_x_mm, target_y_mm,
                               linear_speed=150, angular_speed=150,
                               dt=0.02, pos_tol=8.0, ang_tol=0.02, timeout=15.0):
    """Stop-turn-go: (1) aim, (2) drive straight; stop when within pos_tol, using EKF pose."""
    # 1) Aim
    await turn_toward_point(node, client, ekf_get_state, target_x_mm, target_y_mm,
                            w_cmd=angular_speed, dt=dt, ang_tol=ang_tol, timeout=max(6.0, timeout/3))

    # 2) Drive straight
    await set_motors(node, linear_speed, linear_speed)
    t0 = now()
    while True:
        await client.sleep(dt)  # let motion + EKF update
        x, y, theta = ekf_get_state()[:3]
        dist = math.hypot(target_x_mm - x, target_y_mm - y)

        # Optional slow-down when close
        # if dist < 60.0:
        #     v = max(80, int(linear_speed * 0.6))
        #     await set_motors(node, v, v)

        if dist <= pos_tol:
            break
        if now() - t0 > timeout:
            print("move_to_position_ekf: timeout")
            break
    await stop(node)
    return x, y, theta

async def move_to_position_ekf_vw(node, client, ekf_get_state,
                                  target_x_mm, target_y_mm,
                                  v_cmd=120, kp_heading=250.0,
                                  dt=0.02, pos_tol=8.0, timeout=15.0,
                                  w_clip=200):
    """Smooth V–ω follower (no stop-turn). Uses EKF heading to steer continuously."""
    t0 = now()
    while True:
        x, y, theta = ekf_get_state()[:3]
        dx, dy = target_x_mm - x, target_y_mm - y
        dist = math.hypot(dx, dy)
        if dist <= pos_tol:
            break
        theta_ref = math.atan2(-dx, dy)
        e = wrap_angle(theta_ref - theta)
        w = max(-w_clip, min(w_clip, kp_heading * e))
        # Differential mixing
        await set_motors(node, v_cmd - w, v_cmd + w)
        await client.sleep(dt)
        if now() - t0 > timeout:
            print("move_to_position_ekf_vw: timeout")
            break
    await stop(node)
    return ekf_get_state()[:3]

# -------------------- Path utilities --------------------

def grid_to_mm(path_ij, cell_size_mm):
    # (i=row -> y, j=col -> x)
    return [((j + 0.5) * cell_size_mm, (i + 0.5) * cell_size_mm) for (i, j) in path_ij]

def remove_collinear(pts, eps=1e-9):
    if len(pts) <= 2:
        return pts
    out = [pts[0]]
    for k in range(1, len(pts)-1):
        x0, y0 = out[-1]
        x1, y1 = pts[k]
        x2, y2 = pts[k+1]
        v1 = (x1 - x0, y1 - y0)
        v2 = (x2 - x1, y2 - y0)  # minor improvement: reduce small numerical flicker
        cross = v1[0]*(y2 - y1) - v1[1]*(x2 - x1)
        if abs(cross) > eps:
            out.append((x1, y1))
    out.append(pts[-1])
    return out

# -------------------- EKF-based path follower --------------------

async def follow_astar_path_ekf(node, client, ekf_get_state,
                                path_cells, cell_size_mm=20.0,
                                use_smooth_vw=False,
                                linear_speed=150, angular_speed=150, v_cmd=120, kp_heading=250.0,
                                dt=0.02, pos_tol=8.0, ang_tol=0.02):
    """Follow a path given as grid cells (A*). Converts to mm, simplifies, then visits each waypoint using EKF pose."""
    waypoints = remove_collinear(grid_to_mm(path_cells, cell_size_mm))
    print("Waypoints (mm):", waypoints)

    for (tx, ty) in waypoints:
        if use_smooth_vw:
            _ = await move_to_position_ekf_vw(
                node=node, client=client, ekf_get_state=ekf_get_state,
                target_x_mm=tx, target_y_mm=ty,
                v_cmd=v_cmd, kp_heading=kp_heading,
                dt=dt, pos_tol=pos_tol, timeout=max(15.0, 2.0*pos_tol/ max(1.0, v_cmd)))
        else:
            _ = await move_to_position_ekf(
                node=node, client=client, ekf_get_state=ekf_get_state,
                target_x_mm=tx, target_y_mm=ty,
                linear_speed=linear_speed, angular_speed=angular_speed,
                dt=dt, pos_tol=pos_tol, ang_tol=ang_tol, timeout=15.0)

    # Return final EKF pose for logging
    return ekf_get_state()[:3]