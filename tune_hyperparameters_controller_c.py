from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from config import Config
from neural_network_controllers import ControllerC
from main import train_neural_network


def hypertuning_config(current_config):
    """run a hypertuning config in parallel and return the result"""

    controller_c = ControllerC(hidden_nodes=current_config['hidden_nodes'], num_inputs=Config.NUM_INPUTS, num_outputs=Config.NUM_OUTPUTS)
    max_fitness, success_count = train_neural_network(
        controller_c,
        "Controller C",
        verbose=False,
        mutation_rate=current_config['mutation'],
        crossover_rate=current_config['crossover'],
        tournament_size=current_config['tournament'],
        elitism_count=current_config['elitism'],
        evaluation_trials=1,
    )
    current_config['max_fitness'] = max_fitness
    current_config['success'] = success_count
    return current_config

def main():
    hidden_nodes_options = [5, 10, 15]
    mutation_rates = [0.05, 0.1, 0.2]
    crossover_rates = [0.5, 0.7, 0.9]
    tournament_sizes = [2, 3, 5]
    elitism_counts = [1, 2]

    all_possible_configs = []
    for hidden_node_count in hidden_nodes_options:
        for t_size in tournament_sizes:
            for e_count in elitism_counts:
                for m_rate in mutation_rates:
                    for c_rate in crossover_rates:
                        all_possible_configs.append({
                            'hidden_nodes': hidden_node_count,
                            'mutation': m_rate,
                            'crossover': c_rate,
                            'tournament': t_size,
                            'elitism': e_count,
                        })

    num_workers = os.cpu_count()
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executer:
        futures = {executer.submit(hypertuning_config, current_config): current_config for current_config in all_possible_configs}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
    results.sort(key=lambda x: x['max_fitness'], reverse=True)
    best = results[0]
    print(f"best result hidden_nodes: {best['hidden_nodes']}, mutation: {best['mutation']}, crossover: {best['crossover']}, tournament: {best['tournament']}, elitism: {best['elitism']}, max fitness: {round(best['max_fitness'], 2)}, success: {best['success']}")

if __name__ == "__main__":
    main()
