import math as m
EPSILON = 1e-2
ANGLES_IR_SENSORS = [-m.pi/4, -m.pi/6, 0, m.pi/6, m.pi/4]
SENSORS_WEIGHT = [2, 3, 8, 2, 1]
# Test showing that the sensors are ~1/X where x is distance : 12cm (minimum detection distance) --> 1000   6cm --> 2500 3cm --> 3600
# correction : (12 - distance_to_obstacle)*250 + 1000 = IR_MEASURE (more or less... )
# meaning distance_to_obstacle = 12 - (IR_MEASURE -1000)/250 
def reading_to_distance (read): 
    return 12 - (read-1500)/250

def prox_less_threshold(prox_val : list, threshold: int): 
    for i in prox_val : 
        if i >= threshold : 
            return False
    return True

def vect_calculation (objectif_coord : tuple, curr_pos : tuple, curr_dir : tuple, prox_read : list, REJ_WEIGHT, debug=False) : 
    try : 
        attraction_distance = m.sqrt((objectif_coord[0] - curr_pos[0])**2+ (objectif_coord[1] - curr_pos[1])**2)
        objectif_dir = (objectif_coord[0]- curr_pos[0], objectif_coord[1] - curr_pos [1])
        cross_prod = objectif_dir[1]*curr_dir[0]-objectif_dir[0]*curr_dir[1]
        dot_prod = curr_dir[0]*objectif_dir[0]+objectif_dir[1]*objectif_dir[0]
        angle_to_obj = m.atan2((cross_prod),(dot_prod)) 
        attraction_vect = (m.cos(angle_to_obj)*attraction_distance, m.sin(angle_to_obj)*attraction_distance)# transform from global coordinate to relative coordinates
        rejection_vect = (0,0)
    except ValueError : 
        print(f"dir_vect = {curr_dir}, objectif = {objectif_coord}, curr_pos = {curr_pos}, attraction_norm = {attraction_distance}")
        exit(1)
    for i in range(len(prox_read)): 
        if prox_read[i] > 1500 : 
            rejection_value = SENSORS_WEIGHT[i]/(EPSILON+reading_to_distance(prox_read[i]))
            rejection_vect = (rejection_vect[0]+m.cos(ANGLES_IR_SENSORS[i])*rejection_value, rejection_vect[1]+m.sin(ANGLES_IR_SENSORS[i])*rejection_value)
    if debug : 
        print(f"rejection : {rejection_vect}\nattraction : {attraction_vect}")
    final_vect = (attraction_vect[0] - rejection_vect[0]*REJ_WEIGHT, attraction_vect[1] - rejection_vect[1]*REJ_WEIGHT )
    return final_vect




# Utilities to add global obstacles to repulsion vector in local obstacle avoidance
def add_unknown_repulsion(curr_pos, unknown_cells, UNKNOWN_WEIGHT=0.5, p=1.0, max_range=15.0):
    """
    curr_pos: (x_cm, y_cm)
    unknown_cells: iterable of (x_cm, y_cm) of the centers of -1 cells
    UNKNOWN_WEIGHT: small global weight
    p: distance power for attenuation
    max_range: ignore far unknowns (cm) to limit computation/noise
    """
    rx, ry = curr_pos
    ux, uy = 0.0, 0.0
    for cx, cy in unknown_cells:
        dx = rx - cx
        dy = ry - cy
        d = m.hypot(dx, dy)
        if d < 1e-6 or d > max_range:
            continue
        # unit vector from cell to robot
        ux += (dx / d) * (1.0 / ((d + EPSILON)**p))
        uy += (dy / d) * (1.0 / ((d + EPSILON)**p))
    return (UNKNOWN_WEIGHT * ux, UNKNOWN_WEIGHT * uy)

def unknown_cells_world(grid, cell_size_cm_x, cell_size_cm_y):
    """
    Turn indices of -1 cells into world coordinates (cm).
    origin_world_cm: (x0_cm, y0_cm) of grid cell (0,0)
    res_cm: cell size (cm)
    """
    x0, y0 = (0, 0)
    cells = []
    H, W = len(grid), len(grid[0])
    for i in range(H):
        for j in range(W):
            if grid[i][j] == -1:
                cx = x0 + (j + 0.5) * cell_size_cm_x
                cy = y0 + (i + 0.5) * cell_size_cm_y
                cells.append((cx, cy))
    return cells



