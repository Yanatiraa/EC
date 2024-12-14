import csv
import random
import streamlit as st
import pandas as pd

# Function to read the CSV file and convert it to the desired format
def read_csv_to_dict(file_path):
    program_ratings = {}

    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        # Skip the header
        header = next(reader)

        for row in reader:
            program = row[0]
            ratings = [float(x) for x in row[1:]]  # Convert the ratings to floats
            program_ratings[program] = ratings

    return program_ratings

# Path to the CSV file
file_path = '/content/program_ratings.csv'

# Get the data in the required format
program_ratings_dict = read_csv_to_dict(file_path)

##################################### DEFINING PARAMETERS AND DATASET ################################################################
ratings = program_ratings_dict

GEN = 100  # Number of generations
POP = 50   # Population size
EL_S = 2   # Elitism size

all_programs = list(ratings.keys())  # All programs
all_time_slots = list(range(6, 24))  # Time slots

######################################### DEFINING FUNCTIONS ########################################################################
# Fitness function with weights for prime-time slots
def fitness_function(schedule):
    total_rating = 0
    for time_slot, program in enumerate(schedule):
        weight = 2 if 18 <= time_slot <= 22 else 1  # Prime-time slots get a higher weight
        total_rating += weight * ratings[program][time_slot]
    return total_rating

# Initializing a diverse population
def initialize_population(programs, population_size):
    population = []
    for _ in range(population_size):
        random_schedule = random.sample(programs, len(programs))
        population.append(random_schedule)
    return population

# Crossover
def crossover(schedule1, schedule2):
    crossover_point = random.randint(1, len(schedule1) - 2)
    child1 = schedule1[:crossover_point] + schedule2[crossover_point:]
    child2 = schedule2[:crossover_point] + schedule1[crossover_point:]
    return child1, child2

# Mutation with multi-point mutation
def mutate(schedule):
    mutation_points = random.sample(range(len(schedule)), k=random.randint(1, 3))  # Mutate 1-3 points
    for point in mutation_points:
        schedule[point] = random.choice(all_programs)
    return schedule

# Genetic algorithm
def genetic_algorithm(generations, population_size, crossover_rate, mutation_rate, elitism_size):
    # Initialize population
    population = initialize_population(all_programs, population_size)
    fitness_history = []

    for generation in range(generations):
        # Calculate fitness for the population
        population.sort(key=lambda schedule: fitness_function(schedule), reverse=True)
        fitness_history.append(fitness_function(population[0]))

        # Elitism: Carry forward the best schedules
        new_population = population[:elitism_size]

        while len(new_population) < population_size:
            # Select parents
            parent1, parent2 = random.choices(population[:population_size // 2], k=2)
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            if random.random() < mutation_rate:
                child1 = mutate(child1)
            if random.random() < mutation_rate:
                child2 = mutate(child2)

            new_population.extend([child1, child2])

        population = new_population[:population_size]

    return population[0], fitness_history

##################################### STREAMLIT INTERFACE ###########################################################################
# Streamlit setup
st.title("Scheduling Problem Using Genetic Algorithm")
st.write("Modify the parameters below and view the optimal schedule.")

# User input for parameters
CO_R = st.slider("Crossover Rate (CO_R)", min_value=0.0, max_value=0.95, value=0.8, step=0.01)
MUT_R = st.slider("Mutation Rate (MUT_R)", min_value=0.01, max_value=0.05, value=0.02, step=0.01)

# Run genetic algorithm
final_schedule, fitness_history = genetic_algorithm(
    generations=GEN,
    population_size=POP,
    crossover_rate=CO_R,
    mutation_rate=MUT_R,
    elitism_size=EL_S
)

# Display the results
st.subheader("Optimal Schedule")
schedule_data = {"Time Slot": [f"{time_slot:02d}:00" for time_slot in all_time_slots], "Program": final_schedule}
schedule_df = pd.DataFrame(schedule_data)
st.table(schedule_df)

# Display fitness history
st.subheader("Fitness Over Generations")
st.line_chart(fitness_history)

# Display total ratings
st.write("Total Ratings:", fitness_function(final_schedule))
