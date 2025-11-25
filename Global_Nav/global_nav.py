## A star algorithm since this is the most efficient and I can implement rotation directly in the traveling cost
## input : 2D array with start and end position. Obstacles are mapped as 1 and free is 0 

# Note to self : minimize f(n) = h(n) + g(n) where h is the distance function and g is the movement cost function 
# Movement cost can be computed from orientation, rotation and maybe (?) make it that the cost decreases if we go straight for a while ? Since that would mean we can go "faster" 
# How to define movmement cost ? I think a good way to define movement cost is by pairing it with time. 
# time = velocity/distance, but velocity is a vector, and maybe also considering acceleration ? Or is that too much ? Do I consider that the robot can turn AND move ? 
# Or can the robot only Move OR Turn ? 
import math as m
from heapq import heappop, heappush
EPSILON = 1e-5 
SQRT_2 = m.sqrt(2)
# for debugging only 
import random
from PIL import Image
def get_start(env_map): 
    x,y = 0,0
    for i in env_map:
        for j in i : 
            if j == -2 : 
                return (x,y)
            else :
                y = y+1
        x = x+1 
        y= 0
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


def distance_map(env_map : list[list], start) : 
    # should the start_pos be given in the sense that, should I find it in the matrix or should I take it as parameter ? 
    # Same question for end position
    MAP_SIZE = (len(env_map), len(env_map[0]))
    d_map = []
    for i in range(MAP_SIZE[0]) :
        d_map_row = []
        for j in range(MAP_SIZE[1]): 
            if env_map[i][j] == 1 : 
                env_map[i][j] = -1 
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
    # we consider 90° turn to have the same cost as 1 linear motion 
    g = abs(prev_g[1] - alpha)*4/(m.pi) + linear_cost
    return g
    # TO DO : take into account rotation of the robot

    
def reconstruct_path(current, came_from, start): 
    path = [came_from[current], current]
    previous_pos = came_from[current]
    while previous_pos != start : 
        path.insert(0,came_from[previous_pos])
        previous_pos = came_from[previous_pos]
    return path
# REAL THING : 
def a_star (env_map_orig : list[list], alpha_init) : 
    env_map = env_map_orig.copy()
    MAP_SIZE = (len(env_map), len(env_map[0])) 
    start = get_start(env_map)
    end = get_end(env_map)
    h_map = distance_map(env_map, start)
    open_set = []
    heappush(open_set, (h_map[start[0]][start[1]],0, start)) #heap is (f, g, current)
    came_from = {}
    g_score = {start : (0, alpha_init)}
    while open_set : 
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
                g_score[xplore] = (new_g, alpha)
                f = new_g + h_map[nx][ny]
                heappush(open_set, (f, new_g, xplore))
                came_from[xplore] = current

# debug functions

def debug_generate_maze(rows, cols, obstacle_prob=0.25):
    """
    Generate a random maze of shape (rows x cols).
    -1 = obstacle
    -2 = start
    -3 = end
    0  = free cell
    """

    # Create empty grid
    maze = [[0 for _ in range(cols)] for _ in range(rows)]

    # Random obstacles
    for i in range(rows):
        for j in range(cols):
            if random.random() < obstacle_prob:
                maze[i][j] = -1

    # Pick start and end in free spaces
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
            print("%.2f" % env_map[i][j], end="|")
        print("\n----")
