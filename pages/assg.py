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
# defining fitness function
def fitness_function(schedule):
    total_rating = 0
    for time_slot, program in enumerate(schedule):
        total_rating += ratings[program][time_slot]
    return total_rating

# initializing the population
def initialize_pop(programs, time_slots):
    if not programs:
        return [[]]

    all_schedules = []
    for i in range(len(programs)):
        for schedule in initialize_pop(programs[:i] + programs[i + 1:], time_slots):
            all_schedules.append([programs[i]] + schedule)

    return all_schedules

# selection
def finding_best_schedule(all_schedules):
    best_schedule = []
    max_ratings = 0

    for schedule in all_schedules:
        total_ratings = fitness_function(schedule)
        if total_ratings > max_ratings:
            max_ratings = total_ratings
            best_schedule = schedule

    return best_schedule

# calling the pop func.
all_possible_schedules = initialize_pop(all_programs, all_time_slots)

# callin the schedule func.
best_schedule = finding_best_schedule(all_possible_schedules)

##################################### STREAMLIT INTERFACE ###########################################################################
# Streamlit setup
st.title("Scheduling Problem Using Genetic Algorithm")
st.write("Modify the parameters below and view the optimal schedule.")

# User input for parameters
CO_R = st.slider("Crossover Rate (CO_R)", min_value=0.0, max_value=0.95, value=0.8, step=0.01)
MUT_R = st.slider("Mutation Rate (MUT_R)", min_value=0.01, max_value=0.05, value=0.02, step=0.01)

# Initialize population and find schedules
initial_best_schedule = finding_best_schedule(initialize_pop(all_programs, all_time_slots))
remaining_time_slots = len(all_time_slots) - len(initial_best_schedule)
genetic_schedule = genetic_algorithm(initial_best_schedule, generations=GEN, population_size=POP, crossover_rate=CO_R, mutation_rate=MUT_R, elitism_size=EL_S)

final_schedule = initial_best_schedule + genetic_schedule[:remaining_time_slots]

# Display the results
st.subheader("Optimal Schedule")
schedule_data = {"Time Slot": [f"{time_slot:02d}:00" for time_slot in all_time_slots], "Program": final_schedule}
schedule_df = pd.DataFrame(schedule_data)
st.table(schedule_df)

st.write("Total Ratings:", fitness_function(final_schedule))
