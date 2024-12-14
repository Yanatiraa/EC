import streamlit as st
import csv
import random

# Function to read the CSV file and convert it to the desired format
def read_csv_to_dict(file):
    program_ratings = {}
    # Decode the uploaded file content
    reader = csv.reader(file.read().decode('utf-8').splitlines())
    
    # Skip the header
    header = next(reader)

    for row in reader:
        program = row[0]
        ratings = [float(x) for x in row[1:]]  # Convert the ratings to floats
        program_ratings[program] = ratings

    return program_ratings

# Fitness function
def fitness_function(schedule, ratings):
    total_rating = 0
    for time_slot, program in enumerate(schedule):
        total_rating += ratings[program][time_slot]
    return total_rating

# Initialize population
def initialize_pop(programs, time_slots):
    if not programs:
        return [[]]

    all_schedules = []
    for i in range(len(programs)):
        for schedule in initialize_pop(programs[:i] + programs[i + 1:], time_slots):
            all_schedules.append([programs[i]] + schedule)

    return all_schedules

# Find the best schedule using brute force
def finding_best_schedule(all_schedules, ratings):
    best_schedule = []
    max_ratings = 0

    for schedule in all_schedules:
        total_ratings = fitness_function(schedule, ratings)
        if total_ratings > max_ratings:
            max_ratings = total_ratings
            best_schedule = schedule

    return best_schedule

# Crossover function
def crossover(schedule1, schedule2):
    crossover_point = random.randint(1, len(schedule1) - 2)
    child1 = schedule1[:crossover_point] + schedule2[crossover_point:]
    child2 = schedule2[:crossover_point] + schedule1[crossover_point:]
    return child1, child2

# Mutate a schedule
def mutate(schedule, all_programs):
    mutation_point = random.randint(0, len(schedule) - 1)
    new_program = random.choice(all_programs)
    schedule[mutation_point] = new_program
    return schedule

# Genetic algorithm
def genetic_algorithm(initial_schedule, generations, population_size, crossover_rate, mutation_rate, elitism_size, ratings, all_programs):
    population = [initial_schedule]

    for _ in range(population_size - 1):
        random_schedule = initial_schedule.copy()
        random.shuffle(random_schedule)
        population.append(random_schedule)

    for generation in range(generations):
        new_population = []

        # Elitism
        population.sort(key=lambda schedule: fitness_function(schedule, ratings), reverse=True)
        new_population.extend(population[:elitism_size])

        while len(new_population) < population_size:
            parent1, parent2 = random.choices(population, k=2)
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            if random.random() < mutation_rate:
                child1 = mutate(child1, all_programs)
            if random.random() < mutation_rate:
                child2 = mutate(child2, all_programs)

            new_population.extend([child1, child2])

        population = new_population

    return population[0]

# Streamlit UI elements
st.title('Optimal Program Scheduling with Genetic Algorithm')

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file is not None:
    # Read CSV
    program_ratings_dict = read_csv_to_dict(uploaded_file)
    all_programs = list(program_ratings_dict.keys())  # All programs
    all_time_slots = list(range(6, 24))  # Time slots

    # Default parameters for the genetic algorithm
    GEN = 100       # Number of Generations
    POP = 50        # Population Size
    CO_R = 0.8      # Crossover Rate
    MUT_R = 0.2     # Mutation Rate
    EL_S = 2        # Elitism Size

    # Initialize population and find the best schedule
    all_possible_schedules = initialize_pop(all_programs, all_time_slots)
    initial_best_schedule = finding_best_schedule(all_possible_schedules, program_ratings_dict)

    rem_t_slots = len(all_time_slots) - len(initial_best_schedule)
    genetic_schedule = genetic_algorithm(
        initial_best_schedule,
        generations=GEN,
        population_size=POP,
        crossover_rate=CO_R,
        mutation_rate=MUT_R,
        elitism_size=EL_S,
        ratings=program_ratings_dict,
        all_programs=all_programs
    )

    final_schedule = initial_best_schedule + genetic_schedule[:rem_t_slots]

    # Display the final optimal schedule
    st.subheader("Final Optimal Schedule:")
    for time_slot, program in enumerate(final_schedule):
        st.write(f"Time Slot {all_time_slots[time_slot]:02d}:00 - Program {program}")

    # Display the total ratings
    st.write(f"Total Ratings: {fitness_function(final_schedule, program_ratings_dict)}")
