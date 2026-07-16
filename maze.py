import pybullet as p
import pybullet_data

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
        wall_thick = 1.0

        # outer walls
        self._create_wall(0, 10, 50, wall_thick, wall_height)
        self._create_wall(0, -10, 50, wall_thick, wall_height)
        self._create_wall(-25, 0, wall_thick, 20, wall_height)
        self._create_wall(25, 0, wall_thick, 20, wall_height)

        # maze walls
        self._create_wall(-15, 0, 20, wall_thick, wall_height)
        self._create_wall(15, 3, 20, wall_thick, wall_height)
        self._create_wall(15, -3, 20, wall_thick, wall_height)
        self._create_wall(5, 1, wall_thick, 4, wall_height)

    def _create_wall(self, x, y, length, width, height):
        # create a collusion so that robots would not be able to pass through
        coll_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[length/2, width/2, height/2])
        vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[length/2, width/2, height/2])
        p.createMultiBody(0, coll_id, vis_id, [x, y, height/2])

    def get_random_spawn_location(self):
        """find a random spot where there is no wall"""
        import random
        import math
        while True:
            # should be inside the walls
            random_x = random.uniform(-23, 23)
            random_y = random.uniform(-9, 9)
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

