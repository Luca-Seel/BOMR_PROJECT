import numpy as np
from time import monotonic as now

# Params
BASIC = 80
KP_heading = 250         # to path/waypoint
WALL_DIST = 2500         # desired raw prox value for side sensor
KW_dist = 0.05           # wall distance regulator
KW_tang = 0.6            # tangential fraction of BASIC
FRONT_TH = 2000          # front hit threshold
CLEAR_TH = 1400          # safe clearance to leave wall
MED_TH = 1400            # flank sensor thresholds
STALL_TIMEOUT = 1.0      # s without progress -> wall-follow
PATH_TOL = 40.0          # mm distance to path corridor

state = "go_to_path"
side = +1                # +1 follow obstacle on left (CCW), -1 on right (CW)
last_s_progress = 0.0
t_last_progress = now()

# Debounce memory (last N booleans)
HIT_BUF   = [False, False]   # last 2 frames for "hit"
CLEAR_BUF = [True,  True]    # last 2 frames for "clear"
N_DEB = 2

def distance_point_to_segment(P, A, B):
    # P, A, B are 2D numpy arrays [x, y]
    AB = B - A
    AP = P - A
    denom = AB @ AB
    if denom == 0.0:
        return np.linalg.norm(AP), 0.0, A  # A==B: distance to point A
    t = np.clip((AP @ AB) / denom, 0.0, 1.0)
    proj = A + t * AB
    return np.linalg.norm(P - proj), t, proj

def progress_along_segment(P, A, B):
    # scalar curvilinear abscissa s along segment (0 at A, |AB| at B)
    AB = B - A
    L = np.linalg.norm(AB)
    if L == 0.0:
        return 0.0
    _, t, _ = distance_point_to_segment(P, A, B)
    return t * L

class PathTracker:
    def __init__(self, waypoints, path_tol=40.0, waypoint_tol=50.0):
        # waypoints: list of (x,y) in mm
        self.W = [np.array(w, dtype=float) for w in waypoints]
        self.i = 0
        self.path_tol = path_tol
        self.waypoint_tol = waypoint_tol

    def current_segment(self):
        i = self.i
        if i >= len(self.W) - 1:
            return self.W[-2], self.W[-1]
        return self.W[i], self.W[i+1]

    def update_segment_if_reached(self, pose_xy):
        # advance when close to the next waypoint
        if self.i < len(self.W) - 1:
            _, nxt = self.W[self.i], self.W[self.i+1]
            if np.linalg.norm(pose_xy - nxt) <= self.waypoint_tol:
                self.i = min(self.i + 1, len(self.W) - 2)

    def distance_to_path(self, pose_xy):
        A, B = self.current_segment()
        d, _, _ = distance_point_to_segment(pose_xy, A, B)
        return d

    def progress(self, pose_xy):
        A, B = self.current_segment()
        return progress_along_segment(pose_xy, A, B)

def wrap_angle(theta):
    return (theta + np.pi) % (2*np.pi) - np.pi

def heading_to_waypoint(pose, w_next):
    # returns heading error in rad
    dx, dy = w_next[0] - pose.x, w_next[1] - pose.y
    psi = np.atan2(dy, dx)
    e = wrap_angle(psi - pose.theta)
    return e

def progress_along_path(pose, segment):
    # projection of pose on current path segment abscissa s
    # return scalar s that should increase if advancing
    # (implement with vector projection)
    # pose has pose.x, pose.y
    A, B = segment
    P = np.array([pose.x, pose.y], dtype=float)
    return progress_along_segment(P, A, B)

def go_to_path_ctrl(pose, path_tracker):
    A,B = path_tracker.current_segment()
    P = np.array([pose.x, pose.y]); d, t, proj = distance_point_to_segment(P, A, B)
    d, t, proj = distance_point_to_segment(P, A, B)
    target = proj if d > PATH_TOL/2 else B
    e = heading_to_waypoint(pose, target)  # similar to heading_to_waypoint
    v = BASIC
    w = KP_heading * e
    return v, w

def choose_follow_side(prox):
    # prox array: [front_left, front_center_left, center, front_center_right, front_right, right, left]
    left_strength  = prox[0] + prox[1]
    right_strength = prox[3] + prox[4]
    return -1 if right_strength > left_strength else +1

def follow_wall_ctrl(prox, side):
    # If nothing detected on the 3 central front sensors, don’t steer
    if max(prox[1], prox[2], prox[3]) < 200:  # small noise floor
        v = KW_tang * BASIC
        w = 0.0
        return v, w
    # pick side sensor: left=prox[6], right=prox[5](Thymio%20indexing%20may%20vary)
    side_sensor = prox[1] if side > 0 else prox[3]
    s_mid = prox[2]
    s_flank = prox[3] if side > 0 else prox[1]
    e_dist = (WALL_DIST - side_sensor)  # positive -> too far, negative -> too close
    v_tang = KW_tang * BASIC
    w_lat = side * KW_dist * e_dist     # steer toward desired wall distance
    
    # early turning bias when flank approaches
    bias = 0.0
    if s_flank > MED_TH:
        bias += side * 120.0
    if s_mid > FRONT_TH:
        bias += side * 150.0
    # add small forward bias, reduce if front is too close
    slow = max(0.2, 1.0 - max(0, prox[2]-FRONT_TH)/2500.0)
    v = v_tang * slow
    w = w_lat
    if prox[2] > FRONT_TH:  # strong bias to turn away
        w += side * 150
        v = min(v, 0.6*BASIC)
    print(f"FW side={side} front={prox[:5]} side_sensor={side_sensor} e_dist={e_dist:.0f} w={w:.1f}")
    return v, w

def avoid_reflex(prox):
    # turn away from nearest obstacle
    L = prox[0] + prox[1]
    R = prox[3] + prox[4]
    turn = (R - L) * 0.05        # tune gain
    fslow = max(0.2, 1.0 - prox[2]/3500.0)
    v = BASIC * fslow
    w = np.clip(turn, -200, 200)
    return v, w

def can_rejoin_path(pose, prox, path_tracker):
    # 1) corridor to path/waypoint is clear (front sensors below CLEAR_TH)
    if max(prox[1], prox[2], prox[3]) > CLEAR_TH:
        return False
    # 2) lateral distance to path segment below PATH_TOL
    if path_tracker.distance_to_path(np.array([pose.x, pose.y])) > PATH_TOL:
        return False
    # optional heading check toward next waypoint
    nxt = path_tracker.W[path_tracker.i+1]
    e = abs(heading_to_waypoint(pose, nxt))
    if e > np.deg2rad(60): 
        return False
    # 3) heading not pointing into obstacle (front center below CLEAR_TH)
    return prox[2] < CLEAR_TH

def update_front_flags(prox):
    # raw single-frame tests
    hit_now   = max(prox[1], prox[2], prox[3]) > FRONT_TH
    clear_now = max(prox[1], prox[2], prox[3]) < CLEAR_TH

    # shift buffers (FIFO)
    HIT_BUF.pop(0);   HIT_BUF.append(hit_now)
    CLEAR_BUF.pop(0); CLEAR_BUF.append(clear_now)

    # debounced decisions = all of last N frames agree
    hit_front   = all(HIT_BUF)         # True only if both last frames were "hit"
    clear_front = all(CLEAR_BUF)       # True only if both last frames were "clear"
    return hit_front, clear_front

def control_step(pose, prox, path_tracker):
    global state, side, last_s_progress, t_last_progress
    # detect stall while in go_to_path
    pose_xy = np.array([pose.x, pose.y], dtype=float)
    path_tracker.update_segment_if_reached(pose_xy)
    seg = path_tracker.current_segment()  # tuple (A, B)
    s = progress_along_path(pose, seg)
    hit_front, clear_front = update_front_flags(prox)

    if state == "go_to_path":
        if hit_front:
            side = choose_follow_side(prox)
            state = "follow_wall"
        else:
            if s > last_s_progress + 5.0:  # progressed by 5 mm
                last_s_progress = s
                t_last_progress = now()
            elif now() - t_last_progress > STALL_TIMEOUT:
                side = choose_follow_side(prox)
                state = "follow_wall"

    elif state == "follow_wall":
        #if clear_front and can_rejoin_path(pose, prox, path_tracker):
        if clear_front and can_rejoin_path(pose, prox, path_tracker):
            state = "go_to_path"

    # controls
    if state == "go_to_path":
        print("going back to path")
        v, w = go_to_path_ctrl(pose, path_tracker)
    else:
        v, w = follow_wall_ctrl(prox, side)
        # if other fails use this
        # v, w = avoid_reflex(prox)
        
    # clipping
    w = np.clip(w, -300, 300)
    # convert (v,w) to wheel targets if needed
    # or directly modulate left/right speeds:
    motor_left  = int(np.clip(v - w, -300, 300))
    motor_right = int(np.clip(v + w, -300, 300))
    return motor_left, motor_right, state