
"""

TSM with Tournament selection
- swap mutation
- Partially Mapped Crossover (PMX)

"""

# 1. Imports
import matplotlib.pyplot as plt
import numpy as np
import random
from TSM_supporting_codes import (
    create_initial_population,
    tournament_selection,
    swap_mutation,
    total_distance,
    cities, pmx_crossover
)
from TSM_tournament import (tune_parameters,
                            plot_route,
                            plot_convergence
                            )


param_grid = {
    "population_size": [50, 100, 200],  # Different population sizes
    "mutation_rate": [0.05, 0.1, 0.2],  # Different mutation rates
    "crossover_rate": [0.6, 0.8, 0.9]   # Different crossover probabilities
}




def genetic_algorithm_tsp(cities, population_size=100, generations=500, mutation_rate=0.1, crossover_rate=0.8,
                          stagnation_limit=50):
    """Genetic Algorithm for TSP with early stopping when convergence is reached."""

    population = create_initial_population(cities, population_size)
    best_solution = None
    best_distance = float("inf")
    history = []

    stagnation_counter = 0  # Counts generations with no improvement

    for gen in range(generations):
        new_population = []

        for _ in range(population_size // 2):
            parent1 = tournament_selection(population, cities)
            parent2 = tournament_selection(population, cities)

            # Apply crossover based on Partially Mapped Crossover (PMX)
            if random.random() < crossover_rate:
                child1 = pmx_crossover(parent1, parent2)
                child2 = pmx_crossover(parent2, parent1)
            else:
                child1, child2 = parent1[:], parent2[:]  # No crossover, copy parents

            # Apply mutation
            child1 = swap_mutation(child1, mutation_rate)
            child2 = swap_mutation(child2, mutation_rate)

            new_population.extend([child1, child2])

        population = new_population
        current_best = min(population, key=lambda x: total_distance(x, cities))
        current_best_distance = total_distance(current_best, cities)

        if current_best_distance < best_distance:
            best_solution = current_best
            best_distance = current_best_distance
            stagnation_counter = 0  # Reset stagnation counter since improvement occurred
        else:
            stagnation_counter += 1  # Increment stagnation counter

        history.append(best_distance)

        # Print update every 50 generations
        if gen % 50 == 0:
            print(f"Generation {gen}: Best Distance = {best_distance:.2f}")

        # **Early stopping condition** if no improvement for `stagnation_limit` generations
        if stagnation_counter >= stagnation_limit:
            print(
                f"\nStopping early at generation {gen} due to convergence (no improvement for {stagnation_limit} generations).")
            break  # Exit loop early

    return best_solution, best_distance, history




# Run parameter tuning
best_params, best_distance, best_route, best_history = tune_parameters(cities, param_grid)

# Print the best performing hyperparameters
print(f"\nBest Parameters Found: {best_params}")
print(f"Best Distance Achieved: {best_distance:.2f}")

# Plot the best route found
plot_route(best_route, cities, best_params["population_size"], best_params["mutation_rate"], best_params["crossover_rate"], best_distance)




