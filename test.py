import streamlit as st
import random

st.set_page_config(page_title="Genetic Algorithm")

# Header
st.header("Genetic Algorithm")

# Input fields
mutation_rate = st.number_input("Enter your mutation rate", value=0.2, step=0.01, format="%.2f")
target_name = st.text_input("Enter your name", "Liyana")

# Constants
POP_SIZE = 500
GENES = ' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

# Initialize population
def initialize_pop(target):
    population = []
    for _ in range(POP_SIZE):
        individual = [random.choice(GENES) for _ in range(len(target))]
        population.append(individual)
    return population

# Fitness calculation
def fitness_cal(target, individual):
    return sum(1 for t, i in zip(target, individual) if t != i)

# Selection of top 50% based on fitness
def selection(population, target):
    return sorted(population, key=lambda x: fitness_cal(target, x))[:POP_SIZE // 2]

# Crossover
def crossover(parents, length):
    offspring = []
    for _ in range(POP_SIZE):
        parent1, parent2 = random.sample(parents, 2)
        crossover_point = random.randint(1, length - 1)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        offspring.append(child)
    return offspring

# Mutation
def mutate(population, mutation_rate):
    for individual in population:
        for i in range(len(individual)):
            if random.random() < mutation_rate:
                individual[i] = random.choice(GENES)
    return population

# Main algorithm function
def genetic_algorithm(target, mutation_rate):
    population = initialize_pop(target)
    generation = 1
    found = False
    
    # Display each generation's results iteratively
    output_container = st.container()
    
    while not found:
        # Calculate fitness for the population
        population = [(individual, fitness_cal(target, individual)) for individual in population]
        population.sort(key=lambda x: x[1])
        
        # Display current generation's best result with added spacing
        with output_container:
            st.write(f"String: {''.join(population[0][0])} Generation: {generation} Fitness: {population[0][1]}")
            st.write("")  # Add an empty line for spacing

        # Check if target is reached
        if population[0][1] == 0:
            st.write("Target found")
            break

        # Selection and generation of new offspring
        parents = [individual for individual, _ in population[:POP_SIZE // 2]]
        offspring = crossover(parents, len(target))
        population = mutate(offspring, mutation_rate)
        generation += 1

# Calculate button
if st.button("Calculate"):
    genetic_algorithm(target_name, mutation_rate)
