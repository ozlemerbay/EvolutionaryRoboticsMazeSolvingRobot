# evorob-miniproj2

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

## Control Environment in 3D

I have created test_maze.py file to see the maze and the robot 3D:
```bash
uv run test_maze.py
```
