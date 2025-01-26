import streamlit as st
import random

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
    return random.uniform(0.90, 1.0)

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

    while True:
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

        # Streamlit live output
        st.write(f"Generation {generation}, Best Fitness: {best_fitness:.6f}, Best Individual: {best_individual}")

        if best_fitness >= target_fitness:
            return best_fitness, best_individual

        generation += 1

# Streamlit UI
st.title("Genetic Algorithm for Hyperparameter Optimization")

# User inputs using sliders
target_fitness = st.slider("Target Fitness", min_value=0.90, max_value=1.0, value=0.958, step=0.001)
pop_size = st.slider("Population Size", min_value=10, max_value=200, value=100, step=10)
mut_rate = st.slider("Mutation Rate", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
num_runs = st.slider("Number of Runs", min_value=1, max_value=10, value=1, step=1)

if st.button("Run Genetic Algorithm"):
    with st.spinner("Running Genetic Algorithm..."):
        results = []
        for run in range(1, num_runs + 1):
            st.write(f"**Run {run}**:")
            best_fitness, best_individual = main(pop_size, mut_rate, target_fitness)
            results.append({"Run": run, "Best Fitness": best_fitness, "Best Individual": best_individual})
            st.write(f"Best Fitness: {best_fitness:.6f}")
            st.write(f"Best Individual: {best_individual}")
            st.write("---")
        
        st.success("All runs completed!")
