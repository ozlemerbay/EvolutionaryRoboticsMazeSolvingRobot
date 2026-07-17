import pybullet as p
from maze import Maze
from robot_interface import RobotInterface
from evolution import Evolution
from config import Config
from neural_network_controllers import ControllerA
import matplotlib.pyplot as plt
import math


def run_robot_simulation(maze, robot, robot_id, neural_network, robot_genome, steps, target_position):
    # spawn robot to a random location
    start_pos, angle = maze.get_random_spawn_location()
    p.resetBasePositionAndOrientation(robot_id, start_pos, p.getQuaternionFromEuler([0, 0, angle]))

    neural_network.set_weights(robot_genome)
    for _ in range(steps):
        sensors = robot.get_sensor_data()
        current_position, orientation = p.getBasePositionAndOrientation(robot_id)
        distance_x = target_position[0] - current_position[0]
        distance_y = target_position[1] - current_position[1]
        dist_to_target = math.dist([current_position[0], current_position[1]], [target_position[0], target_position[1]])

        target_angle = math.atan2(distance_y, distance_x)
        _, _, current_angle = p.getEulerFromQuaternion(orientation)
        angle_to_target = target_angle - current_angle
        angle_to_target = (angle_to_target + math.pi) % (2 * math.pi) - math.pi # normalise the angle

        sensors.extend([dist_to_target, angle_to_target])
        left_speed, right_speed = neural_network.forward(sensors)

        # move robot
        # since the speed could only be between -1 and 1, it is too low to move the robot
        # so multiply it with a speed multiplier
        # wheel radius 0.1 x speed 15 = 1.5 meter per second
        robot.set_motor_velocities(left_speed * Config.SPEED_MULTIPLIER, right_speed * Config.SPEED_MULTIPLIER)
        p.stepSimulation()

    end_pos, _ = p.getBasePositionAndOrientation(robot_id)
    return start_pos, end_pos

def calculate_fitness(start_pos, end_pos, target_position, fitness_multiplier):
    start_dist = math.dist([start_pos[0], start_pos[1]], [target_position[0], target_position[1]])
    end_dist = math.dist([end_pos[0], end_pos[1]], [target_position[0], target_position[1]])

    fitness = (start_dist - end_dist) * fitness_multiplier
    return fitness

def save_fitness_plot(mean_history, max_history, min_history, controller_name):
    plt.figure()
    plt.plot(mean_history, label="mean fitness", color="blue")
    plt.plot(max_history, label="max fitness", color="green")
    plt.plot(min_history, label="min fitness", color="red")
    plt.title(f"{controller_name}")
    plt.xlabel("generation")
    plt.ylabel("fitness score")
    plt.legend()
    plot_filename = f"fitness_plot_{controller_name.replace(' ', '_').lower()}.png"
    plt.savefig(plot_filename)
    plt.close()

def train_neural_network(neural_network, controller_name):
    evolution = Evolution(
        population_size=Config.POPULATION_SIZE,
        num_genes=neural_network.total_genes,
        mutation_rate=Config.MUTATION_RATE,
        tournament_size=Config.TOURNAMENT_SIZE,
        crossover_rate=Config.CROSSOVER_RATE,
        elitism_count=Config.ELITISM_COUNT
    )

    mean_history, max_history, min_history = [], [], []

    for gen in range(Config.GENERATIONS):
        maze = Maze()
        maze.init(visualize=False)

        robot_id = p.loadURDF("robot.urdf", basePosition=[0, 0, 0.2])
        robot = RobotInterface(robot_id, sensor_range=Config.SENSOR_RANGE)

        fitness_scores = []
        for robot_genome in evolution.population:
            start_pos, end_pos = run_robot_simulation(maze, robot, robot_id, neural_network, robot_genome, steps=Config.SIMULATION_STEPS, target_position=Config.TARGET_POSITION)
            score = calculate_fitness(start_pos, end_pos, target_position=Config.TARGET_POSITION, fitness_multiplier=Config.FITNESS_MULTIPLIER)
            fitness_scores.append(score)

        gen_mean = sum(fitness_scores) / len(fitness_scores)
        gen_max = max(fitness_scores)
        gen_min = min(fitness_scores)

        mean_history.append(gen_mean)
        max_history.append(gen_max)
        min_history.append(gen_min)

        if gen < Config.GENERATIONS - 1:
            evolution.evolve(fitness_scores)

        p.removeBody(robot_id)
        maze.close()

        print(f"gen {gen} | mean={gen_mean:.2f} | max={gen_max:.2f} | min={gen_min:.2f}")

    save_fitness_plot(mean_history, max_history, min_history, controller_name)

def main():
    controller_a = ControllerA(num_inputs=Config.NUM_INPUTS, num_outputs=Config.NUM_OUTPUTS)
    train_neural_network(controller_a, "Controller A")

if __name__ == "__main__":
    main()
