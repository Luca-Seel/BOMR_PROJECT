import math as m
EPSILON = 1e-2
# Test showing that the sensors are 1/X where x is distance : 12cm (minimum detection distance) --> 1000   6cm --> 2500 3cm --> 3600
# correction : (12 - distance_to_obstacle)*250 + 1000 = IR_MEASURE (more or less... )
# meaning distance_to_obstacle = 12 - (IR_MEASURE -1000)/250 
def reading_to_distance (read): 
    return 12 - (read-1500)/250
ANGLES_IR_SENSORS = [-m.pi/4, -m.pi/8, 0, m.pi/8, m.pi/4]
SENSORS_WEIGHT = [2, 3, 8, 2, 1]
def back_to_global (curr_dir, final_vect):
    relative_angle = m.acos(curr_dir[0]/(m.sqrt(curr_dir[0]**2+curr_dir[1]**2)))
    vect = (final_vect[0]*m.cos(relative_angle)+final_vect[1]*m.sin(relative_angle), final_vect[0]*m.sin(relative_angle)+final_vect[1]*(-m.cos(relative_angle)))
    return vect
def vect_calculation (objectif_coord : tuple, curr_pos : tuple, curr_dir : tuple, prox_read : list, REJ_WEIGHT, debug=False) : 
    try : 
        attraction_distance = m.sqrt((objectif_coord[0] - curr_pos[0])**2+ (objectif_coord[1] - curr_pos[1])**2)
        objectif_dir = (objectif_coord[0]- curr_pos[0], objectif_coord[1] - curr_pos [1])
        angle_to_obj = m.atan2((objectif_dir[1]*curr_dir[0]-objectif_dir[0]*curr_dir[1]),((curr_dir[0]*objectif_dir[0]+objectif_dir[1]*objectif_dir[0])))
        print("angle to abjectif : ", angle_to_obj)
        attraction_vect = (m.cos(angle_to_obj)*attraction_distance, m.sin(angle_to_obj)*attraction_distance)
        rejectoion_vect = (0,0)
    except ValueError : 
        print(f"dir_vect = {curr_dir}, objectif = {objectif_coord}, curr_pos = {curr_pos}, attraction_norm = {attraction_distance}")
        exit(1)
    for i in range(len(prox_read)): 
        if prox_read[i] > 1500 : 
            rejectoion_vect = (rejectoion_vect[0]+SENSORS_WEIGHT[i]*m.cos(ANGLES_IR_SENSORS[i])/(EPSILON+reading_to_distance(prox_read[i])), rejectoion_vect[1]+SENSORS_WEIGHT[i]*m.sin(ANGLES_IR_SENSORS[i])/(EPSILON+reading_to_distance(prox_read[i])))
    if debug : 
        print(f"rejection : {rejectoion_vect}\nattraction : {attraction_vect}")
    final_vect = (attraction_vect[0] - rejectoion_vect[0]*REJ_WEIGHT, attraction_vect[1] - rejectoion_vect[1]*REJ_WEIGHT )
    return final_vect




