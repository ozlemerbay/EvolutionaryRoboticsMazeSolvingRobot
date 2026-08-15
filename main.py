import pybullet as p
from maze import Maze
from robot_interface import RobotInterface
from evolution import Evolution
from config import Config
from neural_network_controllers import ControllerA, ControllerB, ControllerC
import matplotlib.pyplot as plt
import math
import statistics
import random
import numpy as np

# my student id
RANDOM_SEED = 821836

def run_robot_simulation(maze, robot, robot_id, neural_network, robot_genome, steps, target_position):
    # spawn robot to a random location
    start_pos, angle = maze.get_random_spawn_location()
    p.resetBasePositionAndOrientation(robot_id, start_pos, p.getQuaternionFromEuler([0, 0, angle]))

    neural_network.set_weights(robot_genome)
    steps_taken = 0
    collisions = 0
    for _ in range(steps):
        steps_taken += 1
        sensors = robot.get_sensor_data()
        current_position, orientation = p.getBasePositionAndOrientation(robot_id)
        distance_x = target_position[0] - current_position[0]
        distance_y = target_position[1] - current_position[1]
        dist_to_target = math.dist([current_position[0], current_position[1]], [target_position[0], target_position[1]])
        normalized_dist = min(dist_to_target / 30.0, 1.0)

        target_angle = math.atan2(distance_y, distance_x)
        _, _, current_angle = p.getEulerFromQuaternion(orientation)
        angle_to_target = target_angle - current_angle
        angle_to_target = (angle_to_target + math.pi) % (2 * math.pi) - math.pi # normalise the angle
        normalized_angle = angle_to_target / math.pi

        sensors.extend([normalized_dist, normalized_angle])
        left_speed, right_speed = neural_network.forward(sensors)

        # move robot
        # since the speed could only be between -1 and 1, it is too low to move the robot
        # so multiply it with a speed multiplier
        # wheel radius 0.1 x speed 15 = 1.5 meter per second
        robot.set_motor_velocities(left_speed * Config.SPEED_MULTIPLIER, right_speed * Config.SPEED_MULTIPLIER)
        p.stepSimulation()

        # check for collisions
        contact_points = p.getContactPoints(bodyA=robot_id)
        for contact in contact_points:
            if contact[2] != maze.planeId: # if not floor
                collisions += 1
                break

        # early stopping if it reaches the goal
        if dist_to_target < 1.0:
            break

    end_pos, _ = p.getBasePositionAndOrientation(robot_id)
    return start_pos, end_pos, steps_taken, collisions

def calculate_fitness(start_pos, end_pos, target_position, steps_taken, collisions, fitness_multiplier=10):
    dist_needs_to_be_taken = math.dist([start_pos[0], start_pos[1]], [target_position[0], target_position[1]])
    dist_left_to_target = math.dist([end_pos[0], end_pos[1]], [target_position[0], target_position[1]])

    fitness = (dist_needs_to_be_taken - dist_left_to_target) * fitness_multiplier
    fitness -= collisions * 0.1
    fitness -= steps_taken * 0.01

    # give a big bonus for reaching target
    if dist_left_to_target < 1.0:
        fitness += 1000.0

    return fitness

def save_fitness_plot(median_history, max_history, min_history, controller_name):
    plt.figure()
    plt.plot(median_history, label="median fitness", color="blue")
    plt.plot(max_history, label="max fitness", color="green")
    plt.plot(min_history, label="min fitness", color="red")
    plt.title(f"{controller_name}")
    plt.xlabel("generation")
    plt.ylabel("fitness score")
    plt.legend()
    plot_filename = f"fitness_plot_{controller_name.replace(' ', '_').lower()}.png"
    plt.savefig(plot_filename)
    plt.close()

def save_success_plot(success_history, controller_name, max_successes):
    plt.figure()
    plt.plot(success_history, label="reached target", color="purple", marker="o")
    plt.title(f"{controller_name} Success Rate")
    plt.xlabel("generation")
    plt.ylabel("robots reached target")
    plt.ylim(0, max_successes)
    plt.legend()
    plot_filename = f"success_plot_{controller_name.replace(' ', '_').lower()}.png"
    plt.savefig(plot_filename)
    plt.close()

def train_neural_network(neural_network, controller_name, verbose=True, mutation_rate=Config.MUTATION_RATE, crossover_rate=Config.CROSSOVER_RATE, tournament_size=Config.TOURNAMENT_SIZE, elitism_count=Config.ELITISM_COUNT):
    evolution = Evolution(
        population_size=Config.POPULATION_SIZE,
        genome_len=neural_network.genome_len,
        mutation_rate=mutation_rate,
        tournament_size=tournament_size,
        crossover_rate=crossover_rate,
        elitism_count=elitism_count
    )

    median_history, max_history, min_history, success_history = [], [], [], []

    maze = Maze()
    maze.init(visualize=False)

    robot_id = p.loadURDF("robot.urdf", basePosition=[0, 0, 0.2])
    robot = RobotInterface(robot_id, sensor_range=Config.SENSOR_RANGE)

    for generation in range(Config.GENERATIONS):
        success_count = 0
        fitness_scores = []
        for robot_genome in evolution.population:
            start_pos, end_pos, steps_taken, collisions = run_robot_simulation(maze, robot, robot_id, neural_network, robot_genome, steps=Config.SIMULATION_STEPS, target_position=Config.TARGET_POSITION)
            score = calculate_fitness(start_pos, end_pos, target_position=Config.TARGET_POSITION, steps_taken=steps_taken, collisions=collisions, fitness_multiplier=Config.FITNESS_MULTIPLIER)
            dist_left_to_target = math.dist([end_pos[0], end_pos[1]], [Config.TARGET_POSITION[0], Config.TARGET_POSITION[1]])
            # if it reached the target
            if dist_left_to_target < 1.0:
                success_count += 1

            fitness_scores.append(score)

        genome_fitness_score_median = statistics.median(fitness_scores)
        genome_fitness_score_max = max(fitness_scores)
        genome_fitness_score_min = min(fitness_scores)

        median_history.append(genome_fitness_score_median)
        max_history.append(genome_fitness_score_max)
        min_history.append(genome_fitness_score_min)
        success_history.append(success_count)

        if generation < Config.GENERATIONS - 1:
            evolution.evolve(fitness_scores)

        print(f"gen {generation} | median={genome_fitness_score_median:.2f} | max={genome_fitness_score_max:.2f} | min={genome_fitness_score_min:.2f} | reached target: {success_count}/{Config.POPULATION_SIZE}")

    p.removeBody(robot_id)
    maze.close()

    if verbose:
        save_fitness_plot(median_history, max_history, min_history, controller_name)
        save_success_plot(success_history, controller_name, Config.POPULATION_SIZE)

    return max_history[-1], success_history[-1]

def main():
    print(f"Student number as random seed: {RANDOM_SEED}")
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    controller_a = ControllerA(num_inputs=Config.NUM_INPUTS, num_outputs=Config.NUM_OUTPUTS)
    train_neural_network(controller_a, "Controller A", verbose=True, mutation_rate=0.05, crossover_rate=0.7, tournament_size=2, elitism_count=1)
    controller_b = ControllerB(hidden_layers=3, hidden_nodes=15, num_inputs=Config.NUM_INPUTS, num_outputs=Config.NUM_OUTPUTS)
    train_neural_network(controller_b, "Controller B", verbose=True, mutation_rate=0.2, crossover_rate=0.5, tournament_size=5, elitism_count=2)
    controller_c = ControllerC(hidden_nodes=5, num_inputs=Config.NUM_INPUTS, num_outputs=Config.NUM_OUTPUTS)
    train_neural_network(controller_c, "Controller C", verbose=True, mutation_rate=0.05, crossover_rate=0.7, tournament_size=5, elitism_count=2)

if __name__ == "__main__":
    main()
