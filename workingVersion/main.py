from Computer_vision import cv as com
import cv2
import time
import numpy as np
from Global_Nav import global_nav as gb
from tdmclient import ClientAsync, aw

import math as m
from collections import deque

from Filtering import Control_fromEKF as control
from Filtering import Filtering as filt
from Local_Nav import local_nav as ln

async def main(cap):
    
    client = ClientAsync()
    node = await client.wait_for_node() 
    try:
        aw(node.lock()) # lock the node for R/W
    except Exception:
        pass # ignore it it wasn't locked
    
    WAIT_TIME = 0.2
    #---- VISUALIZATION PARAMETERS ----
    # Set camera resolution
    w = 1920
    h = 1080
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_EXPOSURE, -5)
    time.sleep(0.1)  # let the camera apply settings
    
    # Desired display size
    display_width = 960
    display_height = 540
    
    
    #---- INITIALIZATION ---
    ret, frame = cap.read()
    matrix = com.matrix_perspective(frame)
    transformed_frame = com.convert_perspective(frame, matrix)
    
    global_map = com.get_map(transformed_frame)
    robot = com.get_robot(transformed_frame)

    #FIND PATH
    scaling = 0.15
    path = path = gb.a_star(global_map, robot[1], scaling, 
                     (int(round(robot[0][1])), int(round(robot[0][0]))))
    
    #--- NAVIGATION PARAMETERS ---
    # help functions 
    thresh = 120
    def dist_mm(p, q):
        return float(np.hypot(p[0] - q[0], p[1] - q[1]))

    def pixel_to_world_mm(pos):
        px, py = pos
        x = 10 * px * (com.L / com.SIZE[1])
        y = 10 * py * (com.W / com.SIZE[0])
        return x, y
    
    conv_x = 10 * (com.L / com.SIZE[1])
    conv_y = 10 * (com.W / com.SIZE[0])
    # 1) buffers
    traj = deque(maxlen=2000)   # (x,y)
    # convert path
    waypoints = control.remove_collinear(control.grid_to_mm(path, cell_size_mm_x=conv_x, cell_size_mm_y=conv_y))
    step_count = 0
    # ---- Init Kidnap ------
    kidnap_first = False
    kidnap_second = False
    # kidnapping help functions
    async def test_kidnap():
        kidnap_thresh = 500  # off ground
        # Kidnap check
        if np.mean(np.array(node["prox.ground.delta"][:])) < kidnap_thresh :
            print("kidnapped")
            return True
        else:
            return False
    
    def update_kidnap(waypoints, ekf_traj):
        # Compute distance between last two positions
        p1 = ekf_traj[-1][0:2]
        p2 = ekf_traj[-2][0:2]
        # Compute midpoint of the pair
        mean = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        # compute distance to waypoints
        distance_waypoints = [dist_mm(mean, wps) for wps in waypoints]
        min_index = np.argmin(distance_waypoints)
        return min_index
            
    # global obstacle repulsion map
    unk_cells = ln.unknown_cells_world(global_map, conv_x/10, conv_y/10)
    
    # --- INIT EKF----
    # params ekf
    speed_to_mms = 0.4347  # conversion factor from thymio speed units to mm/s 
    # Process noise for EKF (tune) 
    q_proc = (
        1e-10, 1e-10, 1e-3,   # q_x, q_y, q_theta (model mismatch)
        75.72,  0.002692,         # q_v_ctrl, q_omega_ctrl (control execution noise)
        1e-2, 1e-5          # q_v_bias, q_omega_bias (random walk on v, omega)
    )
    # Camera measurement noise 
    r_cam = (1.435, 1.864, 0.001496)  # [mm^2, mm^2, rad^2]
    r_mot = (75.72, 0.002692)    # motor noise on v, omega
    # get first position
    pos, angle, __ = com.get_robot(transformed_frame)
    pos = pixel_to_world_mm(pos)
    x = pos[0]
    y = pos[1]
    x0=[x, y, angle,0,0]
    ekf = filt.EKFState(x0, P0=1000*np.eye(5))
    vl_cmd, vr_cmd = [0, 0]
    
    #--- HELPER FUNCTIONS FOR EKF --- 
    def get_motor_meas(): 
        # raw speeds in Thymio units (instantaneous)
        vl = node["motor.left.speed"]
        vr = node["motor.right.speed"]
        # no IMU because acc is limited
        # convert to v [mm/s], omega [rad/s] 
        v, w = filt.motors_to_vw(vl, vr, speed_to_mms, filt.L)
        print("get_motor_meas", v, w) 
        return np.array([v, w], dtype=float)
    
    def get_cam_meas(image=None):
        # get position from camera
        if image is not None:
            pos, angle, Rob = com.get_robot(image)
            #x = SIZE[1] - pos[0] 
            if Rob ==  False:
                return None
            pos = pixel_to_world_mm(pos)
            x = pos[0]
            y = pos[1]
            #print("Camera Robot Position", x, y, angle)
            return np.array([x, y, angle], dtype=float)
        return None
    
    
    # --- PLOTTING ---
    ekf_traj = [] # for plot
    # memory for plots
    def ekf_get_state():
        s = ekf.get_state()  # (x,y,theta)
        ekf_traj.append((s[0], s[1], s[2]))  # log x,y each time it's called
        return s
    
    #---VISU---
    for i in path : 
        global_map[i[0]][i[1]] = -4 # marker for path in my debug functions
        
    def draw_static_map(global_map, path):
        # Choose a base canvas size matching your transformed frame
        H, W = len(global_map), len(global_map[0])
        
        static = np.full((H, W, 3), 255, dtype=np.uint8)
        
        grown_map = gb.obstacle_scale(global_map.copy(), scaling)
        gm = np.asarray(grown_map)
        obst_y, obst_x = np.where(gm == -1)
        static[obst_y, obst_x] = (0, 255, 255)
        
        gm = np.asarray(global_map)
        obst_y, obst_x = np.where(gm == -1)
        static[obst_y, obst_x] = (0, 0, 255)  # BGR

        # Optional: other map markers (e.g., -3 in blue)
        extra_y, extra_x = np.where(gm == -3)
        static[extra_y, extra_x] = (255, 0, 0)

        # Path (green) drawn once
        for (i, j) in path:
            cv2.circle(static, (j, i), 2, (0, 255, 0), -1)

        return static
    
    static_map_img = draw_static_map(global_map, path)
    
    drawing_robot_real = []
    
    #--- CONNECT TO THYMIO ---
    try:
        aw(node.lock()) # lock the node for R/W
    except Exception:
        pass # ignore it it wasn't locked
    
    aw(node.stop())
    print("Connected:", node)
    motors = aw(node.wait_for_variables({"motor.left.speed","motor.right.speed"}))
    
    # local avoidance
    obstacle_not_passed = False
    stop = False
    
    while True:
        
        #--- GET IMAGE ---
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame")
            break
    
        # Apply perspective transform
        transformed_frame = com.convert_perspective(frame, matrix)
        transformed_frame_visu = transformed_frame.copy() 
        # transformed_frame = current image we can work with
    
    
        #--- VISU ---    
        # Start from cached static image (cheap copy)
        vis = static_map_img.copy()

        # Overlay dynamic info
        robot_px, robot_py, found = None, None, False
        robot, angle, found = com.get_robot(transformed_frame)

        if found:
            # Draw robot arrow
            length = 20
            p0 = (int(robot[0]), int(robot[1]))
            p1 = (int(robot[0] + length*np.cos(angle)),
                int(robot[1] + length*np.sin(angle)))
            cv2.arrowedLine(vis, p0, p1, color=(20, 20, 20), thickness=2, tipLength=0.3)
            
            drawing_robot_real.append(p0)
            
        

        for i in range(len(drawing_robot_real)):
                robot1 = drawing_robot_real[i]
                cv2.circle(vis, robot1, 5, (0,0,255), -1)
        
        for i in range(len(ekf_traj)):
                x, y, _ = ekf_traj[i] # x, y in your EKF state 
                x = com.length_real_to_pixel(x/10, [com.L,com.W], com.SIZE)
                y = com.length_real_to_pixel(y/10, [com.L,com.W], com.SIZE)
                px, py = int(x), int(y) # or apply world→pixel mapping first
                cv2.circle(vis, (px, py), 2, (255, 0, 0), -1)
                
        
        H, W = vis.shape[:2]
        box_h = 48 # banner height in pixels 
        y0 = H - box_h
        status_overlay = vis.copy()
        cv2.rectangle(status_overlay, (0, y0), (W, H), (255, 255, 255), thickness=-1) # white bar
        vis = cv2.addWeighted(status_overlay, 0.6, vis, 0.4, 0.0)
        msgs = []
        if found : msgs.append(("Robot Found", (0, 128, 0))) # green 
        if obstacle_not_passed: msgs.append(("Local obstacle detected", (0, 0, 255))) # red
        if kidnap_first: msgs.append(("KIDNAPPED", (0, 0, 255)))
        x_text = 16 
        y_text = y0 + 30 # baseline inside the bar 
        for k, (msg, color) in enumerate(msgs):
            cv2.putText(vis, msg, (x_text + 250*k, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        
        # Display
        cv2.imshow("Transformed Camera Feed", cv2.resize(vis, (display_width, display_height)))
    
        #--- Camera Window ---
        # Show the transformed frame live
        small_transformed = cv2.resize(transformed_frame_visu, (display_width, display_height))
        cv2.imshow("Transformed Camera", small_transformed)
    
        # Resize for display only
        small_frame = cv2.resize(frame, (display_width, display_height))
    
        # Show the live frame
        cv2.imshow("Live Camera Feed", small_frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            aw(node.set_variables({"motor.left.target":[0], "motor.right.target":[0]}))
            aw(node.unlock())
            break
    
        
        #--- GlOBAL NAVIGATION ---
         # final loop
        if step_count < len(waypoints):
            # get motion params
            z_mot= get_motor_meas()
            z_cam = get_cam_meas(transformed_frame)
            
            # EKF STEP
            if not kidnap_first:
                ekf.step(vl_cmd, vr_cmd, z_cam, r_cam, z_mot, r_mot, filt.Ts, q_proc)
                state = ekf_get_state()
            
            #--- LOCAL AVOIDANCE --- 
            if not(ln.prox_less_threshold(list(node["prox.horizontal"][:5]), 1500)) or obstacle_not_passed :
                print("local avoidance")
                objectif = (waypoints[step_count][0]/10, waypoints[step_count][1]/10) 
                curr_dir = (100*m.cos(state[2]), 100*m.sin(state[2]))
                curr_pos = (state[0]/10, state[1]/10)
                obstacle_not_passed = True
                await node.wait_for_variables({"prox.horizontal"})
                prox_read = list(node["prox.horizontal"])
                
    
                #--- TEST IF OBSTACLED IS PASSED ---
                if (ln.prox_less_threshold(prox_read, 1000)) and obstacle_not_passed:
                    #print("PASSED")
                    aw(control.set_motors(node,100,100))
                    await client.sleep(1.5)
                    obstacle_not_passed = False
                    
                    while step_count < len(waypoints) and dist_mm((state[0], state[1]), waypoints[step_count]) < thresh:
                        step_count += 1
                    continue
                
                vect = ln.vect_calculation(objectif, (curr_pos),curr_dir,  prox_read[:5], 150, debug=False)
                # added for global obstacles
                # Unknown repulsion:
                ux, uy = ln.add_unknown_repulsion(curr_pos, unk_cells, UNKNOWN_WEIGHT=0.25, p=1.0, max_range=15.0)
                # Combine:
                vect = (vect[0] - ux, vect[1] - uy)
                #
                angle_command = m.atan2(vect[1], vect[0])
                delta_speed = np.sign(angle_command)*min(abs(angle_command/WAIT_TIME*800/m.pi), 300)
                left_speed = int(100+(delta_speed/2))
                right_speed = int(100-(delta_speed/2))
                aw(control.set_motors(node, left=left_speed, right=right_speed))
                await client.sleep(WAIT_TIME)
                continue
                
                
            #--- FOLLOW THE GLOBAL PATH ---
            obstacle_not_passed = False
            kidnap_first = aw(test_kidnap())
            if kidnap_first:
                kidnap_second = aw(test_kidnap()) # gives next waypoint to go to if kidnapped
            if not kidnap_first and kidnap_second: # two steps on ground
                # reset filter as last position is false
                for i in range(2) :
                    ret, frame = cap.read()
                    transformed_frame = com.convert_perspective(frame, matrix)
                    pos, angle, __ = com.get_robot(transformed_frame)
                    pos = pixel_to_world_mm(pos)
                    x = pos[0]
                    y = pos[1]
                    x0=[x, y, angle,0,0]
                    ekf.x = x0
                    ekf.P = 1000*np.eye(5)
                    ekf_get_state()
                step_count = update_kidnap(waypoints, ekf_traj)
                kidnap_second = False

            step_count = await control.follow_path(node, state, waypoints, step_count, v_cmd=170, kp_heading=90.0,
                          pos_tol=12.0)
            await client.sleep(0.1)
            
            # stop when goal reached
            if step_count >= len(waypoints):
               stop = True
        vl_cmd, vr_cmd = z_mot # take last values of motor speeds as command for next step     
        
        if stop :
            aw(node.set_variables({"motor.left.target":[0], "motor.right.target":[0]}))
            aw(node.unlock())
            
    
        # quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            aw(node.set_variables({"motor.left.target":[0], "motor.right.target":[0]}))
            aw(node.unlock())
            break
    
