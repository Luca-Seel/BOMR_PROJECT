## A star algorithm since this is the most efficient and I can implement rotation directly in the traveling cost
## input : 2D array with start and end position. Obstacles are mapped as -1 and free is 0 

import math as m
from heapq import heappop, heappush
from scipy.ndimage import distance_transform_edt
import numpy as np
EPSILON = 1e-5 
SQRT_2 = m.sqrt(2) # for computing efficiency
ROBOT_SIZE_CM = 10 #cm 
# for debugging only 
import random
import PIL.Image as Image
def get_end(env_map): 
    x,y = 0,0
    for i in env_map:
        for j in i : 
            if j == -3 : 
                return (x,y)
            else :
                y = y+1
        x = x+1 
        y = 0

def obstacle_scale(env_map : list[list], scale) : 
    ROBOT_SIZE_PX = ROBOT_SIZE_CM/scale 
    obstacle_map = ~(np.array(env_map) == -1)
    distance_to_obstacle_map = distance_transform_edt(obstacle_map) #scpiy function that replaces every non-zero cell with distance to
                                                                    # closest 0 cell
    scaled_obstacles = distance_to_obstacle_map <= ROBOT_SIZE_PX
    return [[-1 if scaled_obstacles[i][j] else env_map[i][j]
             for j in range(len(env_map[0]))]
             for i in range(len(env_map))]

def distance_map(env_map : list[list], start) : 
    # implements heuristic fungction with Euclidian distance
    MAP_SIZE = (len(env_map), len(env_map[0]))
    d_map = []
    for i in range(MAP_SIZE[0]) :
        d_map_row = []
        for j in range(MAP_SIZE[1]): 
            if env_map[i][j] == -1 : 
                d_map_row.append(-1)
            else : 
                d_map_row.append(m.sqrt((start[0]-i)**2 + (start[1]-j)**2))
        d_map.append(d_map_row)
    return d_map

def motion_cost(prev_g, alpha: float) : 
    # start with simple motion cost before anything
    # if diagonal, movment cost is sqrt(2), else 1 
    linear_cost = 0
    if abs(alpha) == m.pi/4 or abs(alpha) == 3*m.pi/4 :
        linear_cost = SQRT_2 + prev_g[0]
    else :
        linear_cost = 1 + prev_g[0]
    # we consider 45° turn to have the same cost as 1 linear motion 
    g = abs(prev_g[1] - alpha)*4/(m.pi) + linear_cost
    return g

    
def reconstruct_path(current, came_from, start): 
    path = [came_from[current], current]
    previous_pos = came_from[current]
    while previous_pos != start : 
        path.insert(0,came_from[previous_pos])
        previous_pos = came_from[previous_pos]
    return path

# A* algorithm with Euclidean distance as h(n) and linear + angle movement cost
def a_star (env_map_orig : list[list], alpha_init, cm_px_scale, start) : 
    MAP_SIZE = (len(env_map_orig), len(env_map_orig[0])) 
    end = get_end(env_map_orig)
    h_map = distance_map(env_map_orig, start)
    env_map = obstacle_scale(env_map_orig.copy(), cm_px_scale)
    open_set = []
    heappush(open_set, (h_map[start[0]][start[1]],0, start)) #heap is (f, g, current)
    came_from = {}
    g_score = {start : (0, alpha_init)}
    counter = 0
    while open_set : 
        counter = counter + 1
        f, g, current = heappop(open_set)
        if current == end : 
            path = reconstruct_path(current, came_from, start)
            return path
        x,y = current
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,1), (1,-1), (-1,-1)] :
            nx, ny = x + dx, y+dy 
            if not((0 <= nx < MAP_SIZE[0]) and (0<= ny < MAP_SIZE[1])) :
                continue
            if env_map[nx][ny] == -1 :
                continue
            alpha = m.atan2(dy, dx)
            new_g = (motion_cost(g_score[current], alpha))
            xplore = (nx,ny)
            if xplore not in g_score or new_g < g_score[xplore][0] : 
                try : 
                    g_score[xplore] = (new_g, alpha)
                    f = new_g + h_map[nx][ny]
                    heappush(open_set, (f, new_g, xplore))
                    came_from[xplore] = current
                except IndexError : 
                    print(f"ALL INDEXES ARE : {nx}, {ny}, {xplore}")
# debug functions
# AI generated to construct random maps and visualize them
def debug_generate_maze(rows, cols, n_blobs=8, blob_size=40):
    """
    Generate a random maze with large obstacle blocks.
    -1 = obstacle
    -2 = start
    -3 = end
    0  = free cell
    
    n_blobs   = how many obstacle clusters to generate
    blob_size = number of steps each blob expands (bigger = larger clusters)
    """

    # Create empty grid
    maze = [[0 for _ in range(cols)] for _ in range(rows)]

    # --- Generate obstacle blobs ---
    for _ in range(n_blobs):
        # Random starting point for the blob
        x = random.randrange(rows)
        y = random.randrange(cols)

        for _ in range(blob_size):
            maze[x][y] = -1  # mark as obstacle

            # Random walk step (blob expansion)
            dx, dy = random.choice([(1,0), (-1,0), (0,1), (0,-1)])
            nx, ny = x + dx, y + dy

            if 0 <= nx < rows and 0 <= ny < cols:
                x, y = nx, ny

    # --- Pick start and end ---
    while True:
        sx, sy = random.randrange(rows), random.randrange(cols)
        if maze[sx][sy] == 0:
            maze[sx][sy] = -2
            break

    while True:
        ex, ey = random.randrange(rows), random.randrange(cols)
        if maze[ex][ey] == 0:
            maze[ex][ey] = -3
            break

    return maze
def debug_maze_to_bitmap(maze, cell_size=20, filename="tests/maze.png"):
    """
    Convert the maze grid into a bitmap PNG image.
    """
    rows = len(maze)
    cols = len(maze[0])

    # Create a blank RGB image
    img = Image.new("RGB", (cols * cell_size, rows * cell_size))
    pixels = img.load()

    # Color mapping
    colors = {
        -1: (0, 0, 0),       # wall
        0: (255, 255, 255),  # free
        -2: (0, 100, 255),   # start
        -3: (255, 40, 40),   # end
        -4: (40, 255, 40),   # Path
    }

    for i in range(rows):
        for j in range(cols):
            c = colors[int(maze[i][j])]
            # Fill the block of size cell_size x cell_size
            for di in range(cell_size):
                for dj in range(cell_size):
                    pixels[j*cell_size + dj, i*cell_size + di] = c

    img.save(filename)
    return img

def print_map(env_map) : 
    MAP_SIZE = (len(env_map), len(env_map[0])) 
    for i in range(MAP_SIZE[1]) :
        for j in range(MAP_SIZE[0]) : 
            print("%.2f" % env_map[j][i], end="|")
        print("\n----")
