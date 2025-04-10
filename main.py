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
def plotObstacles():
    """Plots all obstacles."""
    for poly in obstacles:
        x, y = poly.exterior.xy
        plt.fill(x, y, 'red', alpha=0.5)

def plotRobot(x, y, orientation, color='blue'):
    
    size = 1.0 # length of robot's sides
    corners = np.array([
        [0,0],        # bottom left corner 
        [size, 0],    # bottom right corner
        [size,size],  # top right corner
        [0,size],     # top left corner
    ])
    
    referencePoint = np.array([size / 2, size / 2])
    corners = corners - referencePoint
    
    theta = np.radians(orientation)
    rotationMatrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],                      
    ])
    
    rotatedCorners = np.dot(corners, rotationMatrix.T) #".T" = transpose
    
    translatedCorners = rotatedCorners + np.array([x,y])
    
    plt.fill(*zip(*translatedCorners), color=color, alpha=0.7)
    plt.plot(*zip(*np.vstack([translatedCorners, translatedCorners[0]])), color='black')

    
def distance(p1, p2):
    """Euclidean distance between two points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def isCollisionFree(point):
    """Check if point is inside any obstacle."""
    point = Point(point)
    for poly in obstacles:
        if poly.contains(point):
            return False
    return True

def astar(start, goal):
    """A* pathfinding algorithm."""
    openSet = {tuple(start)}
    cameFrom = {}
    gScore = {tuple(start): 0}
    fScore = {tuple(start): distance(start, goal)}

    while openSet:
        current = min(openSet, key=lambda p: fScore[p])
        if current == tuple(goal):
            # Reconstruct path
            path = []
            while current in cameFrom:
                path.append(current)
                current = cameFrom[current]
            path.append(start)
            path.reverse()
            return path

        openSet.remove(current)
        for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, -1)]:
            neighbor = (current[0] + dx * 0.5, current[1] + dy * 0.5)

            if not isCollisionFree(neighbor):
                continue

            tentativeGScore = gScore[current] + distance(current, neighbor)

            if neighbor not in gScore or tentativeGScore < gScore[neighbor]:
                cameFrom[neighbor] = current
                gScore[neighbor] = tentativeGScore
                fScore[neighbor] = gScore[neighbor] + distance(neighbor, goal)
                openSet.add(neighbor)

    return []

#look thru chatgpt's suggestions for fixing the plot obstacle function
def onClick(event): 
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
    plotObstacles()
    if start:
        plt.scatter(*start, color='green', label='Start')
    if goal:
        plt.scatter(*goal, color='orange', label='Goal')
    plt.legend()
    plt.draw()

def main():
    global orientation
    fig, ax = plt.subplots()

    ax.set_xlim(0, 10) # for some reason i have to use underscores for these
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')

    fig.canvas.mpl_connect('button_press_event', onClick)

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
            plotObstacles()
            plt.plot(*zip(*path), color='blue', label='Path')
            
            # Animate robot along the path
            for i, (x, y) in enumerate(path):
                plt.clf()
                plotObstacles()
                plt.plot(*zip(*path), color='blue', label='Path')
                plotRobot(x, y, orientation)
                plt.legend()
                plt.pause(0.1)

            plt.show()
        else:
            print("No valid path found.")

if __name__ == '__main__':
    main()
