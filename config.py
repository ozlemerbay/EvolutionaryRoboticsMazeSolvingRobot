class Config:
    # evolution hyperparameters
    POPULATION_SIZE = 20
    GENERATIONS = 10
    MUTATION_RATE = 0.1
    CROSSOVER_RATE = 0.7
    TOURNAMENT_SIZE = 3
    ELITISM_COUNT = 1

    # simulation parameters
    SIMULATION_STEPS = 10000
    SENSOR_RANGE = 5.0
    TARGET_POSITION = [8.0, 0.0]
    SPEED_MULTIPLIER = 15.0
    FITNESS_MULTIPLIER = 10.0
    EVALUATION_TRIALS = 3

    # neural network parameters
    NUM_INPUTS = 5
    NUM_OUTPUTS = 2
