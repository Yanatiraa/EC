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
file_path = 'pages/program_ratings.csv'

# Get the data in the required format
program_ratings_dict = read_csv_to_dict(file_path)

##################################### DEFINING PARAMETERS AND DATASET ################################################################
ratings = program_ratings_dict

GEN = 100
POP = 50
EL_S = 2

all_programs = list(ratings.keys())  # All programs
all_time_slots = list(range(6, 24))  # Time slots

######################################### DEFINING FUNCTIONS ########################################################################
# Fitness function
def fitness_function(schedule):
    total_rating = 0
    for time_slot, program in enumerate(schedule):
        total_rating += ratings[program][time_slot]
    return total_rating

# Initialize population
def initialize_pop(programs, time_slots):
    population = []
    for _ in range(POP):
        schedule = random.sample(programs, len(programs))
        population.append(schedule)
    return population

# Selection: Finding the best schedule
def finding_best_schedule(population):
    best_schedule = max(population, key=fitness_function)
    return best_schedule

# Crossover
def crossover(schedule1, schedule2):
    crossover_point = random.randint(1, len(schedule1) - 2)
    child1 = schedule1[:crossover_point] + schedule2[crossover_point:]
    child2 = schedule2[:crossover_point] + schedule1[crossover_point:]
    return child1, child2

# Mutation
def mutate(schedule):
    mutation_point = random.randint(0, len(schedule) - 1)
    new_program = random.choice(all_programs)
    schedule[mutation_point] = new_program
    return schedule

# Genetic Algorithm
def genetic_algorithm(initial_schedule, generations=GEN, population_size=POP, crossover_rate=0.8, mutation_rate=0.2, elitism_size=EL_S):
    population = initialize_pop(all_programs, all_time_slots)

    for generation in range(generations):
        new_population = []

        # Elitism
        population.sort(key=fitness_function, reverse=True)
        new_population.extend(population[:elitism_size])

        while len(new_population) < population_size:
            parent1, parent2 = random.choices(population, k=2)

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

    return finding_best_schedule(population)

##################################### STREAMLIT INTERFACE ###########################################################################
# Streamlit setup
st.title("Scheduling Problem Using Genetic Algorithm")
st.write("Modify the parameters below and view the optimal schedule.")

# User input for parameters
CO_R = st.slider("Crossover Rate (CO_R)", min_value=0.0, max_value=0.95, value=0.8, step=0.01)
MUT_R = st.slider("Mutation Rate (MUT_R)", min_value=0.01, max_value=0.05, value=0.02, step=0.01)

# Initialize population and find schedules
initial_population = initialize_pop(all_programs, all_time_slots)
initial_best_schedule = finding_best_schedule(initial_population)
remaining_time_slots = len(all_time_slots) - len(initial_best_schedule)
genetic_schedule = genetic_algorithm(initial_best_schedule, generations=GEN, population_size=POP, crossover_rate=CO_R, mutation_rate=MUT_R, elitism_size=EL_S)

final_schedule = genetic_schedule[:len(all_time_slots)]

# Display the results
st.subheader("Optimal Schedule")
schedule_data = {"Time Slot": [f"{time_slot:02d}:00" for time_slot in all_time_slots], "Program": final_schedule}
schedule_df = pd.DataFrame(schedule_data)
st.table(schedule_df)

st.write("Total Ratings:", fitness_function(final_schedule))
