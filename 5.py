import random

# Define the target value and the set of numbers
target_value = 10
number_set = [10, 5, 20, 30, 15]

# Genetic Algorithm parameters
population_size = 100
mutation_rate = 0.1
generations = 10

def fitness(individual):
    return abs(sum(individual) - target_value)

def create_individual():
    return [random.choice([0, 1]) for _ in number_set]

def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child

def mutate(individual):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = 1 - individual[i]
    return individual

def genetic_algorithm():
    # Initialize the population
    population = [create_individual() for _ in range(population_size)]

    for generation in range(generations):
        # Evaluate the fitness of each individual in the population
        fitness_scores = [(individual, fitness(individual)) for individual in population]

        # Sort the population by fitness
        population = [individual for individual, _ in sorted(fitness_scores, key=lambda x: x[1])]

        # Select the top individuals for mating
        selected_parents = population[:int(0.2 * population_size)]

        # Create the next generation using crossover and mutation
        children = []
        while len(children) < population_size - len(selected_parents):
            parent1, parent2 = random.choice(selected_parents), random.choice(selected_parents)
            child = crossover(parent1, parent2)
            child = mutate(child)
            children.append(child)

        # Replace the old population with the new generation
        population = selected_parents + children

        # Print the best individual in the current generation
        best_individual = min(fitness_scores, key=lambda x: x[1])[0]
        print(f"Generation {generation + 1}: Best Individual {best_individual} | Fitness: {fitness(best_individual)}")

    # Print the final best individual
    best_individual = min(fitness_scores, key=lambda x: x[1])[0]
    print("\nFinal Result:")
    print(f"Best Individual: {best_individual} | Fitness: {fitness(best_individual)}")

# Run the genetic algorithm
genetic_algorithm()
