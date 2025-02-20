# Genetic Algorithm for Solving the Travelling Salesman Problem (TSP)

### Overview
This repository contains Python scripts and a Jupyter notebook for solving the Travelling Salesman Problem (TSP) using Genetic Algorithms (GA). The implementation includes different selection strategies (Tournament vs. Roulette Selection) and crossover methods (Ordered Crossover (OX1) vs. Partially Mapped Crossover (PMX)) to evaluate their effectiveness in finding optimal solutions.

### Files in the Repository
- `TSM_performance_testing.ipynb` – Jupyter Notebook for testing and visualising different GA configurations.

- `TSM_supporting_codes.py` – Python script containing function definitions for the Genetic Algorithm.

- `TSM_tournament.py` – GA implementation using Tournament Selection with Ordered Crossover (OX1).

- `TSM_tournament_pmx.py` – GA implementation using Tournament Selection with Partially Mapped Crossover (PMX).

- `TSM_roulette.py` – GA implementation using Roulette Selection with Ordered Crossover (OX1).

- `TSM_roulette_pmx.py` – GA implementation using Roulette Selection with Partially Mapped Crossover (PMX).

<br> <!-- Adds a single line break -->

#### Requirements

The following dependencies are required to run the scripts:

```
pip install numpy matplotlib
```

<br> <!-- Adds a single line break -->

### How to Use the Jupyter Notebook

1. Open `TSM_performance_testing.ipynb`.

2. Run all cells step-by-step to generate cities, optimise the TSP using GA, and visualise the results.

3. Modify the hyperparameters (population size, mutation rate, crossover rate) to observe different outcomes.

<br> <!-- Adds a single line break -->

#### Customising the Parameters

The Genetic Algorithm allows the following parameters to be adjusted:

- Population Size: The number of individuals in each generation.

- Mutation Rate: The probability of a mutation occurring in offspring.

- Crossover Rate: The probability of crossover between parents.

- Number of Generations: The total number of iterations for evolution.

To fine-tune these parameters, modify them directly in the scripts:
```
best_route, best_distance, history = genetic_algorithm_tsp(
    cities, population_size=200, generations=500, mutation_rate=0.1
)
```

<br> <!-- Adds a single line break -->

#### Visualizsing Results

1. Best TSP Route: Plots the optimal route found by the GA.

2. Convergence Plot: Displays the best distance over generations to assess convergence speed and effectiveness.

<br> <!-- Adds a single line break -->

### Future Enhancements

- Implementing hybrid optimisation techniques (e.g., Local Search, Simulated Annealing).

- Improving scalability for larger TSP instances.

- Parallelising computations for faster execution.

<br> <!-- Adds a single line break -->

### License

This project is open-source and free to use for research and educational purposes.
