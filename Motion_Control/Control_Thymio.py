import numpy as np
import math
from time import monotonic as now


# Physical constants of the Thymio
WHEELS_DIST_MM = 95.0
SPEED_MM_S = 200.0/500.0 #Calibrate per robot

def wrap_angle(theta):
    return (theta + math.pi) % (2*math.pi) - math.pi

async def set_motors(node, left, right):
    await node.set_variables({
        "motor.left.target": [int(left)],
        "motor.right.target": [int(right)]
    })

async def stop(node):
    await set_motors(node, 0, 0)

async def turn_in_place(node, client, delta_theta, speed=150, dt=0.02, tolerance=0.02, timeout=6.0):
    # sign and targets
    sgn = 1 if delta_theta >= 0 else -1
    await node.set_variables({
        "motor.left.target":  [-sgn*int(speed)],
        "motor.right.target": [ sgn*int(speed)],
    })
    await node.wait_for_variables({"motors.left.speed", "motors.right.speed"})

    theta_acc = 0.0
    t0 = now()
    while True:
        # read instantaneous wheel speeds (Aseba variable names under .v)
        vl = node.v.motors.left.speed
        vr = node.v.motors.right.speed

        # integrate
        dl = vl * SPEED_MM_S * dt
        dr = vr * SPEED_MM_S * dt
        dtheta = (dr - dl) / WHEELS_DIST_MM
        theta_acc += dtheta
        # testing
        print(f"vl={vl} vr={vr} dcenter={0.5*(dl+dr):.2f}")
        #
        # stop if reached
        if abs(delta_theta) - abs(theta_acc) <= tolerance:
            break
        if now() - t0 > timeout:
            print("turn_in_place: timeout")
            break

        await client.sleep(dt)

    await stop(node)
    
async def move_straight(node, client, distance_mm, speed=150, dt=0.02, timeout=10.0):
    dir_sgn = 1 if distance_mm >= 0 else -1
    await set_motors(node, dir_sgn*speed, dir_sgn*speed)
    await node.wait_for_variables({"motors.left.speed", "motors.right.speed"})

    traveled = 0.0
    t0 = now()
    while abs(traveled) < abs(distance_mm):
        vl = node.v.motors.left.speed
        vr = node.v.motors.right.speed
        dl = vl * SPEED_MM_S * dt
        dr = vr * SPEED_MM_S * dt
        dcenter = 0.5 * (dl + dr)
        traveled += dcenter

        if now() - t0 > timeout:
            print("move_straight: timeout")
            break
        await client.sleep(dt)

    await stop(node)

async def move_to_position_odometry(node, client, target_x_mm, target_y_mm,
                                    x=0.0, y=0.0, theta=0.0,
                                    linear_speed=150, angular_speed=150,
                                    dt=0.02, pos_tol=5.0, ang_tol=0.02, timeout=15.0):
    # aim
    dx, dy = target_x_mm - x, target_y_mm - y
    desired_theta = math.atan2(-dx, dy)  # +y forward, +x left
    dth = math.atan2(math.sin(desired_theta - theta), math.cos(desired_theta - theta))
    await turn_in_place(node, client, dth, speed=angular_speed, dt=dt, tolerance=ang_tol)

    # drive
    await set_motors(node, linear_speed, linear_speed)
    await node.wait_for_variables({"motors.left.speed","motors.right.speed"})
    
    t0 = now()
    while True:
        vl = node.v.motors.left.speed
        vr = node.v.motors.right.speed
        dl = vl * SPEED_MM_S * dt
        dr = vr * SPEED_MM_S * dt
        dcenter = 0.5*(dl+dr)
        dtheta  = (dr - dl) / WHEELS_DIST_MM
        theta = wrap_angle(theta + dtheta)
        x += -dcenter*math.sin(theta)
        y +=  dcenter*math.cos(theta)

        # testing
        print(f"vl={vl} vr={vr} dcenter={0.5*(dl+dr):.2f}")
        #
        dist = math.hypot(target_x_mm - x, target_y_mm - y)
        if dist <= pos_tol:
            break
        if now() - t0 > timeout:
            print("move_to_position_odometry: timeout")
            break

        # optional debug (every ~0.5s)
        if int(10*(now() - t0)) % 5 == 0:
            print(f"dist {dist:.1f} mm, vl={vl}, vr={vr}, target v~{linear_speed}")

        await client.sleep(dt)
    
    await stop(node)
    return x, y, theta

# grid cells to mm waypoints; simplify straight segments
def grid_to_mm(path_ij, cell_size_mm):
    return [((j + 0.5) * cell_size_mm, (i + 0.5) * cell_size_mm) for (i, j) in path_ij]

def remove_collinear(pts, eps=1e-9):
    if len(pts) <= 2:
        return pts
    out = [pts[0]]
    for k in range(1, len(pts)-1):
        x0,y0 = out[-1]
        x1,y1 = pts[k]
        x2,y2 = pts[k+1]
        v1 = (x1-x0, y1-y0)
        v2 = (x2-x1, y2-y1)
        # cross product z-component
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        if abs(cross) > eps:
            out.append((x1,y1))
    out.append(pts[-1])
    return out


async def follow_astar_path(node, client, path_cells, x0, y0, th0,
                      cell_size_mm=20.0,
                      linear_speed=150, angular_speed=150,
                      dt=0.01, pos_tol=10.0, ang_tol=0.03):
    # 1) grid to mm
    wps = grid_to_mm(path_cells, cell_size_mm)
    # 2) simplify
    wps = remove_collinear(wps)
    print("Waypoints (mm):", wps)
    x, y, th = x0, y0, th0
    for (tx, ty) in wps:
        print(f"Final pose (odometry): x={x:.1f} mm, y={y:.1f} mm, th={math.degrees(th):.1f} deg")
        x, y, th = await move_to_position_odometry(
            node=node,
            client=client,
            target_x_mm=tx, target_y_mm=ty,
            x=x, y=y, theta=th,
            linear_speed=linear_speed,
            angular_speed=angular_speed,
            dt=dt, pos_tol=pos_tol, ang_tol=ang_tol
        )
    return x, y, th
