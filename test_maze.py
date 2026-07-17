import time
from maze import Maze

for i in range(3):
    maze = Maze()
    maze.init(visualize=True)
    time.sleep(5)
    maze.close()
