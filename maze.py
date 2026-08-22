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
            p.resetDebugVisualizerCamera(cameraDistance=25.0, cameraYaw=0, cameraPitch=-89, cameraTargetPosition=[0, 0, 0])

        self._create_maze()

    def _create_maze(self):
        wall_height = 2.0
        wall_thick = 0.5

        # outer walls (width 37, height 19)
        self._create_wall(-10.5, 9.5, 16.0, wall_thick, wall_height)
        self._create_wall(10.5, 9.5, 16.0, wall_thick, wall_height)
        self._create_wall(0, -9.5, 37.0, wall_thick, wall_height)
        self._create_wall(-18.5, 0, wall_thick, 19.0, wall_height)
        self._create_wall(18.5, 0, wall_thick, 19.0, wall_height)

        # inside walls
        self._create_wall(-10.25, -1.0, 16.5, wall_thick, wall_height)
        self._create_wall(-2.0, -3.5, wall_thick, 5.0, wall_height)
        self._create_wall(11.25, 3.0, 14.5, wall_thick, wall_height)
        self._create_wall(4.0, 1.0, wall_thick, 4.0, wall_height)
        self._create_wall(5.0, -4.0, 9.0, wall_thick, wall_height)

    def _create_wall(self, x, y, length, width, height):
        # create a collusion so that robots would not be able to pass through
        coll_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[length/2, width/2, height/2])
        vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[length/2, width/2, height/2])
        p.createMultiBody(0, coll_id, vis_id, [x, y, height/2])

    def get_random_spawn_location(self):
        """find a random spot where there is no wall for robot to spawn"""
        true_random = random.SystemRandom() # do not use seed so that it would be really random
        while True:
            # spawn robot randomly inside the maze
            random_x = true_random.uniform(-17.0, 17.0)
            random_y = true_random.uniform(-8.0, 8.0)

            robot_size_box_min = [random_x - 1.2, random_y - 1.2, 0.1]
            robot_size_box_max = [random_x + 1.2, random_y + 1.2, 0.5]
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
        random_angle = true_random.uniform(0, 2 * math.pi)
        return [random_x, random_y, 0.2], random_angle

    def close(self):
        p.disconnect()

