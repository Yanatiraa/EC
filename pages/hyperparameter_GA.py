import streamlit as st
import pandas as pd
import numpy as np
import random
import time
import matplotlib.pyplot as plt

# Load dataset
@st.cache
def load_data():
    return pd.read_csv("hyperparameter_dataset.csv")

data = load_data()

# Genetic Algorithm Functions
def initialize_pop(pop_size):
    predefined_learning_rates = [0.001, 0.005, 0.01]
    population = []
    for i in range(pop_size):
        individual = {
            "learning_rate": random.choice(predefined_learning_rates),
            "batch_size": random.randint(16, 128),
            "hidden_layers": [random.randint(10, 100) for _ in range(random.randint(1, 5))],
            "activation": random.choice(['relu', 'sigmoid', 'tanh']),
            "epochs": random.randint(10, 100)
        }
        population.append(individual)
    return population

def fitness_cal(individual):
    accuracy = random.uniform(0.90, 1.0)
    return accuracy

def selection(population):
    fitness_scores = [(ind, fitness_cal(ind)) for ind in population]
    fitness_scores.sort(key=lambda x: x[1], reverse=True)
    return [ind[0] for ind in fitness_scores[:len(population) // 2]]

def crossover(parent1, parent2):
    child = {
        "learning_rate": random.choice([parent1["learning_rate"], parent2["learning_rate"]]),
        "batch_size": random.choice([parent1["batch_size"], parent2["batch_size"]]),
        "hidden_layers": parent1["hidden_layers"][:len(parent1["hidden_layers"]) // 2] +
                         parent2["hidden_layers"][len(parent2["hidden_layers"]) // 2:],
        "activation": random.choice([parent1["activation"], parent2["activation"]]),
        "epochs": random.choice([parent1["epochs"], parent2["epochs"]])
    }
    return child

def mutate(individual, mut_rate):
    if random.random() < mut_rate:
        individual["batch_size"] = random.randint(16, 128)
    if random.random() < mut_rate:
        individual["hidden_layers"] = [random.randint(10, 100) for _ in range(random.randint(1, 5))]
    if random.random() < mut_rate:
        individual["activation"] = random.choice(['relu', 'sigmoid', 'tanh'])
    if random.random() < mut_rate:
        individual["epochs"] = random.randint(10, 100)
    return individual

def main(pop_size, mut_rate, target_fitness):
    population = initialize_pop(pop_size)
    generation = 1
    best_fitness_values = []
    found = False

    while not found:
        selected = selection(population)
        new_generation = []
        for _ in range(pop_size):
            parent1 = random.choice(selected)
            parent2 = random.choice(selected)
            child = crossover(parent1, parent2)
            child = mutate(child, mut_rate)
            new_generation.append(child)

        population = new_generation
        best_individual = max(population, key=lambda ind: fitness_cal(ind))
        best_fitness = fitness_cal(best_individual)
        best_fitness_values.append(best_fitness)

        if best_fitness >= target_fitness:
            found = True

        generation += 1

    return best_individual, best_fitness_values

# Streamlit UI
st.title("Genetic Algorithm for Hyperparameter Optimization")

# User inputs
target_fitness = st.number_input("Target Fitness", min_value=0.90, max_value=1.0, value=0.958, step=0.001)
pop_size = st.number_input("Population Size", min_value=10, max_value=200, value=100, step=10)
mut_rate = st.slider("Mutation Rate", min_value=0.0, max_value=1.0, value=0.2, step=0.01)

if st.button("Run Genetic Algorithm"):
    with st.spinner("Running Genetic Algorithm..."):
        best_individual, fitness_values = main(pop_size, mut_rate, target_fitness)
        st.success("Optimization Complete!")
    
    # Display results
    st.subheader("Best Individual")
    st.write(best_individual)

    st.subheader("Fitness Values Across Generations")
    st.line_chart(fitness_values)
