# Evolutionary Robotics - Autonomous Maze Navigation
**Student Number (Problem ID / Seed): 821836**

## Hyperparameter Tuning
I have created hyperparameter tuning files for each controller since they have different structures.
Tuning took too long, as a result I left them for a run in the evenings and saved the results to txt files automatically using the tee command.
Here are the commands to run each tuning process:

For controller_a:
```bash
uv run tune_hyperparameters_controller_a.py | tee tuned_hyperparameters_controller_a.txt
```

For controller_b:
```bash
uv run tune_hyperparameters_controller_b.py | tee tuned_hyperparameters_controller_b.txt
```

For controller_c:
```bash
uv run tune_hyperparameters_controller_c.py | tee tuned_hyperparameters_controller_c.txt
```

## Run trainings
To run the full evolutionary training for all three controllers, reproduce the results, and automatically generate all the fitness and success .png plots, run the following command:
```bash
uv run main.py | tee results.txt
```

## Showcasing Trained Models
Best performing weights for each controller are saved into `.npy` files. These files (`best_genome_controller_a.npy`, `best_genome_controller_b.npy`, and `best_genome_controller_c.npy`).
I have created the `deploy_models.py` file to demonstrate how the trained models work. This script loads the saved `.npy` models and show how they solve the maze. To see it run the following command:
```bash
uv run deploy_models.py
```
