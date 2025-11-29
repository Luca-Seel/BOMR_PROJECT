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

async def move_to_pos(node, state, target_x_mm, target_y_mm,
                v_cmd=20, kp_heading=250.0, w_clip=20):
    x, y, theta = state[:3]
    print("ekf:", x,y,theta)
    dx, dy = target_x_mm - x, target_y_mm - y
    print("dx, dy:", dx, dy)
    theta_ref = math.atan2(dy, dx)
    e = wrap_angle(theta_ref - theta)
    w = max(-w_clip, min(w_clip, kp_heading * e))
    if abs(e) > np.pi/8:
        await set_motors(node, w, -w)
        print("motor sets:", w, -w)
    else:
        await set_motors(node, v_cmd + w, v_cmd - w)
        print("motor sets:", v_cmd + w, v_cmd - w)
    
    #await stop(node)
    return dx, dy
        
# -------------------- Path utilities --------------------

def grid_to_mm(path_ij, cell_size_mm_x, cell_size_mm_y):
    # (i=row -> y, j=col -> x)
    return [((j + 0.5) * cell_size_mm_x, (i + 0.5) * cell_size_mm_y) for (i, j) in path_ij]

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

async def follow_path(node, state, waypoints, v_cmd=200, kp_heading=250.0,
                      pos_tol=8.0):
    """Follow a path given as waypoints (A*)."""
    tx, ty = waypoints[0]
    dx, dy = await move_to_pos(node, state, tx, ty,
                v_cmd=v_cmd, kp_heading=kp_heading, w_clip=200)
    # check if waypoint reached, then delete it from our list
    dist = math.hypot(dx, dy)
    print("dist:", dist)
    if dist <= pos_tol:
        waypoints.pop(0) # remove first
    return waypoints