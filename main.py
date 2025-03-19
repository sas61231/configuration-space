import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from scipy.spatial import KDTree
import math

# --- Global Variables ---
obstacles = []
start = None
goal = None
orientation = 0  # In degrees

# --- Helper Functions ---
def plot_obstacles():
    """Plots all obstacles."""
    for poly in obstacles:
        x, y = poly.exterior.xy
        plt.fill(x, y, 'red', alpha=0.5)

def plot_robot(x, y, orientation, color='blue'):
    """Plots the robot with orientation."""
    size = 0.3
    dx = size * np.cos(np.radians(orientation))
    dy = size * np.sin(np.radians(orientation))

    # Plot the robot as a triangle
    robot_shape = np.array([
        [x, y],
        [x + dx, y + dy],
        [x - dy, y + dx]
    ])
    plt.plot(*zip(*np.vstack([robot_shape, robot_shape[0]])), color=color)

def distance(p1, p2):
    """Euclidean distance between two points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def is_collision_free(point):
    """Check if point is inside any obstacle."""
    point = Point(point)
    for poly in obstacles:
        if poly.contains(point):
            return False
    return True

def astar(start, goal):
    """A* pathfinding algorithm."""
    open_set = {tuple(start)}
    came_from = {}
    g_score = {tuple(start): 0}
    f_score = {tuple(start): distance(start, goal)}

    while open_set:
        current = min(open_set, key=lambda p: f_score[p])
        if current == tuple(goal):
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        open_set.remove(current)
        for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, -1)]:
            neighbor = (current[0] + dx * 0.5, current[1] + dy * 0.5)

            if not is_collision_free(neighbor):
                continue

            tentative_g_score = g_score[current] + distance(current, neighbor)

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + distance(neighbor, goal)
                open_set.add(neighbor)

    return []

def on_click(event):
    """Handles mouse clicks for placing obstacles, start, and goal."""
    global start, goal

    if event.button == 1:  # Left-click: Place obstacle vertex
        if len(obstacles) == 0 or len(obstacles[-1].exterior.coords) > 3:
            obstacles.append(Polygon([]))
        obstacles[-1] = Polygon(list(obstacles[-1].exterior.coords) + [(event.xdata, event.ydata)])
    elif event.button == 2:  # Middle-click: Set start point
        start = (event.xdata, event.ydata)
    elif event.button == 3:  # Right-click: Set goal point
        goal = (event.xdata, event.ydata)

    plt.clf()
    plot_obstacles()
    if start:
        plt.scatter(*start, color='green', label='Start')
    if goal:
        plt.scatter(*goal, color='orange', label='Goal')
    plt.legend()
    plt.draw()

def main():
    global orientation
    fig, ax = plt.subplots()

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')

    fig.canvas.mpl_connect('button_press_event', on_click)

    print("Controls:")
    print("- Left-click: Add obstacle points")
    print("- Middle-click: Set start point")
    print("- Right-click: Set goal point")
    print("- Press Enter to start path planning")

    plt.show()

    # Path Planning
    if start and goal:
        path = astar(start, goal)
        
        if path:
            plt.figure()
            plt.title("Path Planning")
            plot_obstacles()
            plt.plot(*zip(*path), color='blue', label='Path')
            
            # Animate robot along the path
            for i, (x, y) in enumerate(path):
                plt.clf()
                plot_obstacles()
                plt.plot(*zip(*path), color='blue', label='Path')
                plot_robot(x, y, orientation)
                plt.legend()
                plt.pause(0.1)

            plt.show()
        else:
            print("No valid path found.")

if __name__ == '__main__':
    main()
