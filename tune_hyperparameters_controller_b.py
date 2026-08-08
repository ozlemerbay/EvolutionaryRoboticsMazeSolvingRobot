from config import Config
from neural_network_controllers import ControllerB
from main import train_neural_network

def main():
    hidden_layers = [1, 2, 3]
    mutation_rates = [0.05, 0.1, 0.2]
    crossover_rates = [0.5, 0.7, 0.9]
    tournament_sizes = [2, 3, 5]
    elitism_counts = [1, 2]

    results = []
    for hidden_layer_count in hidden_layers:
        for t_size in tournament_sizes:
            for e_count in elitism_counts:
                for m_rate in mutation_rates:
                    for c_rate in crossover_rates:
                        controller_b = ControllerB(hidden_layers=hidden_layer_count, num_inputs=Config.NUM_INPUTS, num_outputs=Config.NUM_OUTPUTS)
                        max_fitness, success_count = train_neural_network(
                            controller_b,
                            "Controller B",
                            verbose=False,
                            mutation_rate=m_rate,
                            crossover_rate=c_rate,
                            tournament_size=t_size,
                            elitism_count=e_count,
                            evaluation_trials=1
                        )
                        results.append({
                            'hidden_layers': hidden_layer_count,
                            'mutation': m_rate,
                            'crossover': c_rate,
                            'tournament': t_size,
                            'elitism': e_count,
                            'max_fitness': max_fitness,
                            'success': success_count
                        })
    #get best result
    results.sort(key=lambda x: x['max_fitness'], reverse=True)
    best = results[0]
    print(f"best result hidden_layers: {best['hidden_layers']}, mutation: {best['mutation']}, crossover: {best['crossover']}, tournament: {best['tournament']}, elitism: {best['elitism']}, max fitness: {round(best['max_fitness'], 2)}, success: {best['success']}")

if __name__ == "__main__":
    main()
