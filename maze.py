import pybullet as p
import pybullet_data
import random
import math

class Maze:
    def init(self, visualize=False):
        self.physicsClient = p.connect(p.GUI if visualize else p.DIRECT)
        p.setGravity(0, 0, -9.8)
        # set the path for pybullet_data
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # load the plane
        self.planeId = p.loadURDF("plane.urdf")

        if visualize:
            p.resetDebugVisualizerCamera(cameraDistance=12.0, cameraYaw=0, cameraPitch=-89, cameraTargetPosition=[0, 0, 0])

        self._create_maze()

    def _create_maze(self):
        wall_height = 2.0
        wall_thick = 0.5

        # outer walls
        self._create_wall(0, 10, 20, wall_thick, wall_height)
        self._create_wall(0, -10, 20, wall_thick, wall_height)
        self._create_wall(-10, 0, wall_thick, 20, wall_height)
        self._create_wall(10, 0, wall_thick, 20, wall_height)

        # grid is 5x5 see maze.png
        ROWS = 5
        COLS = 5

        visited = [[False for _ in range(COLS)] for _ in range(ROWS)]
        vertical_walls = [[True for _ in range(COLS-1)] for _ in range(ROWS)]
        horizontal_walls = [[True for _ in range(COLS)] for _ in range(ROWS-1)]
        wall_list = []

        start_r = random.randint(0, ROWS-1)
        start_c = random.randint(0, COLS-1)
        visited[start_r][start_c] = True

        def add_walls(r, c):
            if c > 0: wall_list.append(('v', r, c-1))
            if c < COLS-1: wall_list.append(('v', r, c))
            if r > 0: wall_list.append(('h', r-1, c))
            if r < ROWS-1: wall_list.append(('h', r, c))
        add_walls(start_r, start_c)

        # use Prim's Algorithm
        while wall_list:
            wall_index = random.randint(0, len(wall_list)-1)
            w_type, r, c = wall_list.pop(wall_index)

            if w_type == 'v':
                cell1 = (r, c)
                cell2 = (r, c+1)
            else:
                cell1 = (r, c)
                cell2 = (r+1, c)

            v1 = visited[cell1[0]][cell1[1]]
            v2 = visited[cell2[0]][cell2[1]]

            # if there is a wall between visited and unvisited cell, remove it
            if v1 != v2:
                if w_type == 'v':
                    vertical_walls[r][c] = False
                else:
                    horizontal_walls[r][c] = False

                if not v1:
                    visited[cell1[0]][cell1[1]] = True
                    add_walls(cell1[0], cell1[1])
                else:
                    visited[cell2[0]][cell2[1]] = True
                    add_walls(cell2[0], cell2[1])

        # create vertical walls
        for r in range(ROWS):
            for c in range(COLS-1):
                if vertical_walls[r][c]:
                    x = -10 + (c + 1) * 4
                    y = -10 + r * 4 + 2
                    self._create_wall(x, y, wall_thick, 4 + wall_thick, wall_height)

        # create horizontal walls
        for r in range(ROWS-1):
            for c in range(COLS):
                if horizontal_walls[r][c]:
                    x = -10 + c * 4 + 2
                    y = -10 + (r + 1) * 4
                    self._create_wall(x, y, 4 + wall_thick, wall_thick, wall_height)

    def _create_wall(self, x, y, length, width, height):
        # create a collusion so that robots would not be able to pass through
        coll_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[length/2, width/2, height/2])
        vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[length/2, width/2, height/2])
        p.createMultiBody(0, coll_id, vis_id, [x, y, height/2])

    def get_random_spawn_location(self):
        """find a random spot where there is no wall"""
        while True:
            # spawn robot in the left side away from the target
            random_x = random.uniform(-8.0, -8.0)
            random_y = random.uniform(-8.0, 8.0)

            robot_size_box_min = [random_x - 0.3, random_y - 0.3, 0.1]
            robot_size_box_max = [random_x + 0.3, random_y + 0.3, 0.5]
            overlapping = p.getOverlappingObjects(robot_size_box_min, robot_size_box_max)

            is_safe = True
            if overlapping:
                for obj in overlapping:
                    body_id = obj[0]
                    # if not floor then wall
                    if body_id != self.planeId:
                        is_safe = False
                        break
            if is_safe:
                break
        # pick random angle
        random_angle = random.uniform(0, 2 * math.pi)
        return [random_x, random_y, 0.2], random_angle

    def close(self):
        p.disconnect()

