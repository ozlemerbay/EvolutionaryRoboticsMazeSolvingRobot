import time
import pybullet as p
from maze import Maze

def main():
    maze = Maze()
    maze.init(visualize=True)


    robot_id = p.loadURDF("robot.urdf", basePosition=[0, 0, 0.2])
    start_pos, angle = maze.get_random_spawn_location()
    p.resetBasePositionAndOrientation(robot_id, start_pos, p.getQuaternionFromEuler([0, 0, angle]))

    try:
        while True:
            p.stepSimulation()
            time.sleep(1.0 / 240.0)
    except KeyboardInterrupt:
        pass
    finally:
        maze.close()

if __name__ == "__main__":
    main()
