
"""

TSM with Tournament selection
- swap mutation
- ordered crossover

"""



# 1. Imports

import numpy as np
import matplotlib.pyplot as plt
import random
import itertools
from TSM_supporting_codes import (
                                tune_parameters,
                                create_initial_population,
                                tournament_selection,
                                ordered_crossover,
                                swap_mutation,
                                total_distance,
                                cities
                                )

param_grid = {
    "population_size": [50, 100, 200],  # Different population sizes
    "mutation_rate": [0.05, 0.1, 0.2],  # Different mutation rates
    "crossover_rate": [0.6, 0.8, 0.9]   # Different crossover probabilities
}

import matplotlib.pyplot as plt
import numpy as np


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

            # Apply crossover based on crossover_rate
            if random.random() < crossover_rate:
                child1 = ordered_crossover(parent1, parent2)
                child2 = ordered_crossover(parent2, parent1)
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



def plot_convergence(history, population_size, mutation_rate, crossover_rate, best_distance):
    """Plots the convergence of the GA and includes parameters in the title for easy comparison."""
    plt.figure(figsize=(8, 4))
    plt.plot(history, label=f"Pop: {population_size}, Mut: {mutation_rate}, Cross: {crossover_rate}")

    plt.xlabel("Generations")
    plt.ylabel("Best Distance")
    plt.title(
        f"GA Convergence\nPop: {population_size}, Mut: {mutation_rate}, Cross: {crossover_rate}, Best Dist: {best_distance:.2f}")
    plt.legend()
    plt.grid()
    plt.show()


def plot_route(route, cities, population_size, mutation_rate, crossover_rate, best_distance):
    """Plots the best TSP route and includes parameters in the title."""
    plt.figure(figsize=(8, 6))
    ordered_cities = np.array([cities[i] for i in route] + [cities[route[0]]])  # Close the loop
    plt.plot(ordered_cities[:, 0], ordered_cities[:, 1], 'o-', label="Route")

    for i, city in enumerate(cities):
        plt.text(city[0], city[1], str(i), fontsize=12, ha="right")

    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title(
        f"Best TSP Route\nPop: {population_size}, Mut: {mutation_rate}, Cross: {crossover_rate}, Best Dist: {best_distance:.2f}")
    plt.legend()
    plt.grid()
    plt.show()



def tune_parameters(cities, param_grid, generations=500, stagnation_limit=50):
    best_params = None
    best_distance = float("inf")
    best_history = []
    best_route = None

    # Iterate over all parameter combinations
    for pop_size in param_grid["population_size"]:
        for mutation_rate in param_grid["mutation_rate"]:
            for crossover_rate in param_grid["crossover_rate"]:

                # Print test configuration **before** running GA, with flush=True
                print(f"\n--- Testing: Population={pop_size}, Mutation={mutation_rate}, Crossover={crossover_rate} ---",
                      flush=True)

                # Run the GA with early stopping
                best_route_run, distance, history = genetic_algorithm_tsp(
                    cities, population_size=pop_size, generations=generations,
                    mutation_rate=mutation_rate, crossover_rate=crossover_rate, stagnation_limit=stagnation_limit
                )

                # Plot convergence for this run
                plot_convergence(history, pop_size, mutation_rate, crossover_rate, distance)

                # Update best parameters if a better solution is found
                if distance < best_distance:
                    best_distance = distance
                    best_params = {
                        "population_size": pop_size,
                        "mutation_rate": mutation_rate,
                        "crossover_rate": crossover_rate
                    }
                    best_history = history
                    best_route = best_route_run

    return best_params, best_distance, best_route, best_history



# Run parameter tuning
best_params, best_distance, best_route, best_history = tune_parameters(cities, param_grid, stagnation_limit=50)

# Print the best performing hyperparameters
print(f"\nBest Parameters Found: {best_params}")
print(f"Best Distance Achieved: {best_distance:.2f}")

# Plot the best route found
plot_route(best_route, cities, best_params["population_size"], best_params["mutation_rate"], best_params["crossover_rate"], best_distance)

