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
    # print("I set some motor values in set_motor")
    #print(int(left))
    #print(int(right))
    #print(int(node.v.motor.left.speed), int(node.v.motor.right.speed))

async def stop(node):
    await set_motors(node, 0, 0)

# -------------------- Go-to-waypoint primitives (EKF) --------------------

async def move_to_position_ekf_vw(node, client, ekf_get_state,
                                  target_x_mm, target_y_mm,
                                  v_cmd=200, kp_heading=250.0,
                                  dt=0.02, pos_tol=8.0, timeout=15.0,
                                  w_clip=20):
    """Smooth V–ω follower (no stop-turn). Uses EKF heading to steer continuously."""
    t0 = now()
    while True:
        x, y, theta = ekf_get_state()[:3]
        print("ekf results in move to pos:")
        print(x, y, theta)
        dx, dy = target_x_mm - x, target_y_mm - y
        dist = math.hypot(dx, dy)
        if dist <= pos_tol:
            break
        theta_ref = math.atan2(dx, dy)
        e = wrap_angle(theta_ref - theta)
        w = max(-w_clip, min(w_clip, kp_heading * e))
        # Differential mixing
        print("motor left")
        print(v_cmd-w)
        print("motor right")
        print(v_cmd+w)
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
                                path_cells, cell_size_mm=2.0,
                                use_smooth_vw=False,
                                linear_speed=50, angular_speed=20, v_cmd=200, kp_heading=250.0,
                                dt=0.02, pos_tol=8.0, ang_tol=0.02):
    """Follow a path given as grid cells (A*). Converts to mm, simplifies, then visits each waypoint using EKF pose."""
    waypoints = remove_collinear(grid_to_mm(path_cells, cell_size_mm))
    print("Waypoints (mm):", waypoints)

    for (tx, ty) in waypoints:
        print("doing waypoint:")
        print(tx, ty)
        _ = await move_to_position_ekf_vw(
            node=node, client=client, ekf_get_state=ekf_get_state,
            target_x_mm=tx, target_y_mm=ty,
            v_cmd=v_cmd, kp_heading=kp_heading,
            dt=dt, pos_tol=pos_tol, timeout=max(15.0, 2.0*pos_tol/ max(1.0, v_cmd)))
    # Return final EKF pose for logging
    return ekf_get_state()[:3]