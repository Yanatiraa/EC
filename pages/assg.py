import streamlit as st
import csv
import random

##################################### DATA LOADING FUNCTION ################################################################
# Function to read the CSV file and convert it to the desired format
def read_csv_to_dict(file_path):
    program_ratings = {}
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        header = next(reader)  # Skip the header
        for row in reader:
            program = row[0]
            ratings = [float(x) for x in row[1:]]  # Convert the ratings to floats
            program_ratings[program] = ratings
    return program_ratings

# Path to the CSV file
file_path = 'program_ratings.csv'  # Update this path as needed
program_ratings_dict = read_csv_to_dict(file_path)

##################################### PARAMETERS AND DATASET ################################################################
# Initial parameters (default values)
GEN = 100
POP = 50
DEFAULT_CO_R = 0.8
DEFAULT_MUT_R = 0.2
EL_S = 2

ratings = program_ratings_dict
all_programs = list(ratings.keys())  # All programs
all_time_slots = list(range(6, 24))  # Time slots

##################################### GENETIC ALGORITHM FUNCTIONS ################################################################
# Fitness function
def fitness_function(schedule):
    total_rating = 0
    for time_slot, program in enumerate(schedule):
        total_rating += ratings[program][time_slot]
    return total_rating

# Initialize population
def initialize_pop(programs, time_slots):
    all_schedules = []
    for i in range(len(programs)):
        for schedule in initialize_pop(programs[:i] + programs[i + 1:], time_slots):
            all_schedules.append([programs[i]] + schedule)
    return all_schedules

# Find best schedule
def finding_best_schedule(all_schedules):
    best_schedule = []
    max_ratings = 0
    for schedule in all_schedules:
        total_ratings = fitness_function(schedule)
        if total_ratings > max_ratings:
            max_ratings = total_ratings
            best_schedule = schedule
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
def genetic_algorithm(initial_schedule, generations=GEN, population_size=POP, crossover_rate=DEFAULT_CO_R, mutation_rate=DEFAULT_MUT_R, elitism_size=EL_S):
    population = [initial_schedule]
    for _ in range(population_size - 1):
        random_schedule = initial_schedule.copy()
        random.shuffle(random_schedule)
        population.append(random_schedule)
    for generation in range(generations):
        new_population = []
        population.sort(key=lambda schedule: fitness_function(schedule), reverse=True)
        new_population.extend(population[:elitism_size])  # Elitism
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
        population = new_population
    return population[0]

##################################### STREAMLIT INTEGRATION ################################################################
st.title("Genetic Algorithm Scheduling")
st.sidebar.header("Algorithm Parameters")

# Sidebar inputs for Crossover Rate and Mutation Rate
CO_R = st.sidebar.slider("Crossover Rate", 0.0, 0.95, DEFAULT_CO_R)
MUT_R = st.sidebar.slider("Mutation Rate", 0.01, 0.05, DEFAULT_MUT_R)

if st.sidebar.button("Run Genetic Algorithm"):
    initial_best_schedule = finding_best_schedule(initialize_pop(all_programs, all_time_slots))
    rem_t_slots = len(all_time_slots) - len(initial_best_schedule)
    final_schedule = genetic_algorithm(initial_best_schedule, crossover_rate=CO_R, mutation_rate=MUT_R)
    final_schedule = initial_best_schedule + final_schedule[:rem_t_slots]

    # Display the schedule
    st.subheader("Optimal Schedule")
    schedule_table = [{"Time Slot": f"{all_time_slots[i]:02d}:00", "Program": final_schedule[i]} for i in range(len(final_schedule))]
    st.table(schedule_table)

    # Display the total ratings
    st.subheader("Total Ratings")
    st.write(f"Total Ratings: {fitness_function(final_schedule)}")
