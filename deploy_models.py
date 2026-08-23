import pybullet as p
from maze import Maze
from robot_interface import RobotInterface
from config import Config
from neural_network_controllers import ControllerA, ControllerB, ControllerC
import math
import time
import numpy as np
import random

RANDOM_SEED = 821836

def run_single_simulation(maze, robot, robot_id, neural_network, genome, steps, target_position, visualize=False):
    start_pos, angle = maze.get_random_spawn_location()
    p.resetBasePositionAndOrientation(robot_id, start_pos, p.getQuaternionFromEuler([0, 0, angle]))
    neural_network.set_weights(genome)

    for _ in range(steps):
        sensors = robot.get_sensor_data()
        current_position, orientation = p.getBasePositionAndOrientation(robot_id)

        distance_x = target_position[0] - current_position[0]
        distance_y = target_position[1] - current_position[1]
        dist_to_target = math.dist([current_position[0], current_position[1]], [target_position[0], target_position[1]])
        normalized_dist = min(dist_to_target / 30.0, 1.0)

        target_angle = math.atan2(distance_y, distance_x)
        _, _, current_angle = p.getEulerFromQuaternion(orientation)
        angle_to_target = target_angle - current_angle
        angle_to_target = (angle_to_target + math.pi) % (2 * math.pi) - math.pi
        normalized_angle = angle_to_target / math.pi

        sensors.extend([normalized_dist, normalized_angle])
        left_speed, right_speed = neural_network.forward(sensors)

        robot.set_motor_velocities(left_speed * Config.SPEED_MULTIPLIER, right_speed * Config.SPEED_MULTIPLIER)
        p.stepSimulation()

        # did not use it because this is makes robot very slow
        # but without this, it looks like robot is teleporting
        # if visualize:
        #    time.sleep(1/50000000)

        if dist_to_target < 1.0:
            break

    end_pos, _ = p.getBasePositionAndOrientation(robot_id)
    return dist_to_target

def deploy():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    maze_visual = Maze()
    maze_visual.init(visualize=True)
    robot_id_visual = p.loadURDF("robot.urdf", basePosition=[0, 0, 0.2])
    robot_visual = RobotInterface(robot_id_visual, sensor_range=Config.SENSOR_RANGE)

    controllers = [
        ("Controller A", ControllerA(num_inputs=Config.NUM_INPUTS, num_outputs=Config.NUM_OUTPUTS)),
        ("Controller B", ControllerB(hidden_layers=3, hidden_nodes=15, num_inputs=Config.NUM_INPUTS, num_outputs=Config.NUM_OUTPUTS)),
        ("Controller C", ControllerC(hidden_nodes=5, num_inputs=Config.NUM_INPUTS, num_outputs=Config.NUM_OUTPUTS))
    ]

    for name, neural_network in controllers:
        model_filename = f"best_genome_{name.replace(' ', '_').lower()}.npy"
        try:
            genome = np.load(model_filename)
            for attempt in range(5):
                dist = run_single_simulation(maze_visual, robot_visual, robot_id_visual, neural_network, genome, Config.SIMULATION_STEPS, Config.TARGET_POSITION, visualize=True)
                if dist < 1.0:
                    print(f"Target reached successfully using {name}.")
                else:
                    print(f"Target not reached using {name}.")
                time.sleep(1)
        except FileNotFoundError:
            print(f"Model file not found.")

    maze_visual.close()

if __name__ == "__main__":
    deploy()
