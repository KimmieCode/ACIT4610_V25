
# 1. Imports

import numpy as np
import matplotlib.pyplot as plt
import random
import itertools

# 2. Defines TSP
def euclidean_distance(city1, city2):
    return np.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)

def total_distance(route, cities):
    return sum(euclidean_distance(cities[route[i]], cities[route[i + 1]]) for i in range(len(route) - 1)) + \
           euclidean_distance(cities[route[-1]], cities[route[0]])  # Return to start city


# 3. Initialise the population

def create_initial_population(cities, population_size):
    population = []
    for _ in range(population_size):
        individual = list(np.random.permutation(len(cities)))  # Random permutation of city indices
        population.append(individual)
    return population


# 4. Define the fitness function

def fitness(individual, cities):
    return 1 / total_distance(individual, cities)  # Lower distance -> Higher fitness



# 5. Define selection

def tournament_selection(population, cities, tournament_size=5):
    selected = random.sample(population, tournament_size)
    selected.sort(key=lambda x: fitness(x, cities), reverse=True)
    return selected[0]  # Best individual


def roulette_wheel_selection(population, cities):
    fitness_values = [fitness(ind, cities) for ind in population]
    total_fitness = sum(fitness_values)
    selection_probs = [f / total_fitness for f in fitness_values]

    return population[np.random.choice(len(population), p=selection_probs)]



# 6. Defines crossover

def ordered_crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))  # Select two random crossover points

    child = [None] * size
    child[start:end] = parent1[start:end]  # Copy a segment from parent1

    # Fill in the remaining cities from parent2
    fill_values = [city for city in parent2 if city not in child]
    fill_index = 0
    for i in range(size):
        if child[i] is None:
            child[i] = fill_values[fill_index]
            fill_index += 1

    return child


def pmx_crossover(parent1, parent2):
    size = len(parent1)
    child = [None] * size
    start, end = sorted(random.sample(range(size), 2))

    # Copy a slice from parent1
    child[start:end] = parent1[start:end]

    # Fill in remaining values from parent2
    for i in range(start, end):
        if parent2[i] not in child:
            val = parent2[i]
            while val in child:
                val = parent2[parent1.index(val)]
            child[i] = val

    # Fill remaining None positions
    for i in range(size):
        if child[i] is None:
            child[i] = parent2[i]

    return child


# 7. Defines swap mutation

def swap_mutation(individual, mutation_rate=0.1):
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(individual)), 2)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual



# 8. Defines the GA TSP algorithm

def genetic_algorithm_tsp(cities, population_size=100, generations=500, mutation_rate=0.1):
    # Step 1: Create Initial Population
    population = create_initial_population(cities, population_size)

    best_solution = None
    best_distance = float("inf")
    history = []  # Track best fitness over generations

    for gen in range(generations):
        new_population = []

        # Step 2: Create Next Generation
        for _ in range(population_size // 2):  # Produce population_size offspring
            parent1 = tournament_selection(population, cities)
            parent2 = tournament_selection(population, cities)

            # Crossover
            child1 = ordered_crossover(parent1, parent2)
            child2 = ordered_crossover(parent2, parent1)

            # Mutation
            child1 = swap_mutation(child1, mutation_rate)
            child2 = swap_mutation(child2, mutation_rate)

            new_population.extend([child1, child2])

        population = new_population

        # Step 3: Track Best Solution
        current_best = min(population, key=lambda x: total_distance(x, cities))
        current_best_distance = total_distance(current_best, cities)

        if current_best_distance < best_distance:
            best_solution = current_best
            best_distance = current_best_distance

        history.append(best_distance)

        if gen % 50 == 0:
            print(f"Generation {gen}: Best Distance = {best_distance:.2f}")

    return best_solution, best_distance, history











# 9. Running the GA TSP

# Generate Random Cities
num_cities = 20
cities = np.random.rand(num_cities, 2) * 100  # Random coordinates in a 100x100 grid

# Run GA
best_route, best_distance, history = genetic_algorithm_tsp(cities)


# Visualizing the Best Route
def plot_route(route, cities, title="Best TSP Route"):
    plt.figure(figsize=(8, 6))
    ordered_cities = np.array([cities[i] for i in route] + [cities[route[0]]])  # Close the loop
    plt.plot(ordered_cities[:, 0], ordered_cities[:, 1], 'o-', label="Route")

    for i, city in enumerate(cities):
        plt.text(city[0], city[1], str(i), fontsize=12, ha="right")

    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title(title)
    plt.legend()
    plt.show()


plot_route(best_route, cities)

# Visualizing Convergence
plt.figure(figsize=(8, 4))
plt.plot(history)
plt.xlabel("Generations")
plt.ylabel("Best Distance")
plt.title("Convergence Plot")
plt.show()


# 10. hyper-tuning parameters

def tune_parameters(cities, param_grid):
    best_params = None
    best_distance = float("inf")

    for pop_size in param_grid["population_size"]:
        for mutation_rate in param_grid["mutation_rate"]:
            for crossover_rate in param_grid["crossover_rate"]:
                print(f"Testing: Population={pop_size}, Mutation={mutation_rate}, Crossover={crossover_rate}")

                # Run GA with current parameters
                best_route, distance, _ = genetic_algorithm_tsp(
                    cities, population_size=pop_size, generations=500, mutation_rate=mutation_rate
                )

                if distance < best_distance:
                    best_distance = distance
                    best_params = {"population_size": pop_size, "mutation_rate": mutation_rate, "crossover_rate": crossover_rate}

    return best_params, best_distance
