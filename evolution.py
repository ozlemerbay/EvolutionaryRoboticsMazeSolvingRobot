import random
import numpy as np
class Evolution:
    def __init__(self, population_size, genome_len, mutation_rate, tournament_size, crossover_rate, elitism_count):
        self.population_size = population_size
        self.genome_len = genome_len
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        self.population = []

        # create random population
        for _ in range(population_size):
            genome = np.random.uniform(-1.0, 1.0, genome_len)
            self.population.append(genome)

    def _tournament_selection(self, population_with_scores, tournament_size):
        """return the best individual from the random selected group of population"""
        tournament = random.sample(population_with_scores, tournament_size)
        tournament.sort(key=lambda x: x[1], reverse=True)
        return tournament[0][0]

    def _crossover(self, parent1, parent2):
        """apply uniform crossover"""
        if random.random() < self.crossover_rate:
            child = np.zeros(self.genome_len)
            for i in range(self.genome_len):
                if random.random() < 0.5:
                    child[i] = parent1[i]
                else:
                    child[i] = parent2[i]
            return child
        else:
            return parent1.copy()

    def _mutate(self, genome):
        """apply uniform mutation"""
        for i in range(self.genome_len):
            if random.random() < self.mutation_rate:
                genome[i] += random.uniform(-0.5, 0.5)
        return genome

    def evolve(self, fitness_scores):
        population_with_scores = list(zip(self.population, fitness_scores))
        population_with_scores.sort(key=lambda x: x[1], reverse=True)
        new_population = []

        for i in range(self.elitism_count):
            if i < len(population_with_scores):
                new_population.append(population_with_scores[i][0].copy())

        while len(new_population) < self.population_size:
            parent_1 = self._tournament_selection(population_with_scores, self.tournament_size)
            parent_2 = self._tournament_selection(population_with_scores, self.tournament_size)
            child = self._crossover(parent_1, parent_2)
            child = self._mutate(child)
            new_population.append(child)

        self.population = new_population
