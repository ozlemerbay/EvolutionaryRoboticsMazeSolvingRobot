from config import Config
from neural_network_controllers import ControllerA
from main import train_neural_network

def main():
    mutation_rates = [0.05, 0.1, 0.2]
    crossover_rates = [0.5, 0.7, 0.9]

    results = []
    for m_rate in mutation_rates:
        for c_rate in crossover_rates:
            Config.MUTATION_RATE = m_rate
            Config.CROSSOVER_RATE = c_rate
            controller_a = ControllerA(num_inputs=Config.NUM_INPUTS, num_outputs=Config.NUM_OUTPUTS)
            max_fitness, success_count = train_neural_network(controller_a, "Controller A", verbose=False)
            results.append({
                'mutation': m_rate,
                'crossover': c_rate,
                'max_fitness': max_fitness,
                'success': success_count
            })
    #get best result
    results.sort(key=lambda x: x['max_fitness'], reverse=True)
    best = results[0]
    print(f"best result mutation: {best['mutation']}, crossover: {best['crossover']}, max fitness: {round(best['max_fitness'], 2)}, success: {best['success']}")

if __name__ == "__main__":
    main()
