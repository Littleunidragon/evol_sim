import random
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
import time
import sqlite3
import numpy as np

# Constants (default values)
DEFAULT_POPULATION_SIZE = 20
DEFAULT_GENE_LENGTH = 20  # Increased gene length for more diversity
DEFAULT_MUTATION_RATE = 0.1
DEFAULT_NUM_GENERATIONS = 50

# Neural network parameters
INPUT_SIZE = 3  # Input layer size
HIDDEN_SIZE = 5  # Hidden layer size
OUTPUT_SIZE = 1  # Output layer size

# Function to generate a random creature with a neural network
def generate_creature():
    creature = {}
    creature['weights_input_hidden'] = np.random.randn(INPUT_SIZE, HIDDEN_SIZE)
    creature['weights_hidden_output'] = np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE)
    return creature

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Function to calculate the output of the neural network
def calculate_output(inputs, creature):
    hidden_layer_output = sigmoid(np.dot(inputs, creature['weights_input_hidden']))
    output = sigmoid(np.dot(hidden_layer_output, creature['weights_hidden_output']))
    return output

# Function to calculate the fitness of a creature
def calculate_fitness(creature):
    # Example fitness function: sum of the weights
    return np.sum(creature['weights_input_hidden']) + np.sum(creature['weights_hidden_output'])

# Function to perform crossover between two creatures
def crossover(creature1, creature2):
    child1 = {}
    child2 = {}
    child1['weights_input_hidden'] = np.zeros_like(creature1['weights_input_hidden'])
    child1['weights_hidden_output'] = np.zeros_like(creature1['weights_hidden_output'])
    child2['weights_input_hidden'] = np.zeros_like(creature2['weights_input_hidden'])
    child2['weights_hidden_output'] = np.zeros_like(creature2['weights_hidden_output'])
    for i in range(INPUT_SIZE):
        for j in range(HIDDEN_SIZE):
            if random.random() < 0.5:
                child1['weights_input_hidden'][i][j] = creature1['weights_input_hidden'][i][j]
                child2['weights_input_hidden'][i][j] = creature2['weights_input_hidden'][i][j]
            else:
                child1['weights_input_hidden'][i][j] = creature2['weights_input_hidden'][i][j]
                child2['weights_input_hidden'][i][j] = creature1['weights_input_hidden'][i][j]
    for i in range(HIDDEN_SIZE):
        for j in range(OUTPUT_SIZE):
            if random.random() < 0.5:
                child1['weights_hidden_output'][i][j] = creature1['weights_hidden_output'][i][j]
                child2['weights_hidden_output'][i][j] = creature2['weights_hidden_output'][i][j]
            else:
                child1['weights_hidden_output'][i][j] = creature2['weights_hidden_output'][i][j]
                child2['weights_hidden_output'][i][j] = creature1['weights_hidden_output'][i][j]
    return child1, child2

# Function to perform mutation on a creature
def mutate(creature):
    mutation_rate = 0.1
    for i in range(INPUT_SIZE):
        for j in range(HIDDEN_SIZE):
            if random.random() < mutation_rate:
                creature['weights_input_hidden'][i][j] += np.random.randn()
    for i in range(HIDDEN_SIZE):
        for j in range(OUTPUT_SIZE):
            if random.random() < mutation_rate:
                creature['weights_hidden_output'][i][j] += np.random.randn()

# Function to select parents based on the chosen method
def select_parents(population, fitness_values, method):
    if method == 'Roulette Wheel Selection':
        return roulette_wheel_selection(population, fitness_values)
    elif method == 'Tournament Selection':
        return tournament_selection(population, fitness_values)

# Roulette Wheel Selection method
def roulette_wheel_selection(population, fitness_values):
    total_fitness = sum(fitness_values)
    roulette_wheel = [fitness / total_fitness for fitness in fitness_values]
    parent1_index = random.choices(range(len(population)), weights=roulette_wheel)[0]
    parent2_index = random.choices(range(len(population)), weights=roulette_wheel)[0]
    return population[parent1_index], population[parent2_index]

# Tournament Selection method
def tournament_selection(population, fitness_values):
    tournament_size = min(3, len(population))  # Tournament size capped at population size
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitness = [fitness_values[i] for i in tournament_indices]
    parent1_index = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitness = [fitness_values[i] for i in tournament_indices]
    parent2_index = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
    return population[parent1_index], population[parent2_index]

# Main evolution function
def evolve(population_size, gene_length, mutation_rate, num_generations, parent_selection_method):
    # Connect to the SQLite database
    conn = sqlite3.connect('evolution_data.db')
    c = conn.cursor()

    # Create evolution_results table if not exists
    c.execute('''CREATE TABLE IF NOT EXISTS evolution_results
                 (generation INTEGER, best_creature TEXT, fitness REAL)''')

    # Generate initial population
    population = [generate_creature() for _ in range(population_size)]

    # Lists to store data for plotting
    best_fitnesses = []

    # Evolution loop
    for generation in range(num_generations):
        # Calculate fitness for each creature
        fitness_values = [calculate_fitness(creature) for creature in population]

        # Select parents and perform crossover to create new generation
        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = select_parents(population, fitness_values, parent_selection_method)
            child1, child2 = crossover(parent1, parent2)
            mutate(child1)
            mutate(child2)
            new_population.extend([child1, child2])

        # Replace the old population with the new one
        population = new_population

        # Record best creature's fitness for plotting
        best_creature = max(population, key=calculate_fitness)
        best_fitnesses.append(calculate_fitness(best_creature))

        # Insert generation data into the database
        c.execute("INSERT INTO evolution_results VALUES (?, ?, ?)", (generation + 1, str(best_creature), calculate_fitness(best_creature)))

        # Print best creature in each generation
        print(f"Generation {generation + 1}: Best creature - {best_creature}, Fitness - {calculate_fitness(best_creature)}")

    # Commit changes and close database connection
    conn.commit()
    conn.close()

    # Plot the evolution of the best creature's fitness
    plt.plot(range(1, num_generations + 1), best_fitnesses)
    plt.xlabel('Generation')
    plt.ylabel('Best Creature Fitness')
    plt.title('Evolution of Best Creature Fitness')
    plt.grid(True)
    plt.show()

# Function to start evolution when the Start button is clicked
def start_evolution():
    population_size = int(population_size_entry.get())
    gene_length = int(gene_length_entry.get())  # Get gene length from input field
    mutation_rate = float(mutation_rate_entry.get())
    num_generations = int(num_generations_entry.get())  # Get number of generations from input field
    parent_selection_method = parent_selection_method_combobox.get()
    evolve(population_size, gene_length, mutation_rate, num_generations, parent_selection_method)

# Function to calculate statistics across multiple runs
def calculate_overall_statistics():
    # Connect to the SQLite database
    conn = sqlite3.connect('evolution_data.db')
    c = conn.cursor()

    # Retrieve fitness values from the database
    c.execute("SELECT fitness FROM evolution_results")
    fitness_values = [row[0] for row in c.fetchall()]

    # Calculate overall statistics
    avg_fitness = np.mean(fitness_values)
    std_dev_fitness = np.std(fitness_values)

    # Close database connection
    conn.close()

    # Display statistics
    overall_statistics_label.config(text=f"Overall Avg Fitness: {avg_fitness:.2f}, Overall Std Dev Fitness: {std_dev_fitness:.2f}")

# Function to display the explanation window
def show_explanation():
    if not hasattr(show_explanation, "explanation_window"):  # Check if the window is already open
        show_explanation.explanation_window = tk.Toplevel(root)
        show_explanation.explanation_window.title("Explanation")
        show_explanation.explanation_window.geometry("400x300+100+100")  # Set position to (100, 100)
        
        # Create a frame to hold the text and scrollbar
        frame = ttk.Frame(show_explanation.explanation_window)
        frame.pack(fill=tk.BOTH, expand=True)

        # Add a scrollbar
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Add a text widget
        explanation_text = """
        Evolution Simulation Explanation:

        Population Size: The number of creatures in each generation.

        Gene Length: The length of the genetic code for each creature.

        Mutation Rate: The probability that a gene will mutate (change) during reproduction.

        Number of Generations: The total number of generations the simulation will run.

        Generation: Each iteration of the evolutionary process. In each generation, creatures reproduce and pass on their genetic traits to the next generation.

        Best Fit Creature: The creature with the highest fitness (i.e., the sum of its genetic code) in a generation.

        Fitness: A measure of how well-adapted a creature is to its environment. In this simulation, it's the sum of the creature's genetic code.

        Crossover: The process by which genetic material is exchanged between two parent creatures to produce offspring.

        Mutation: A random change in the genetic code of a creature.

        Parent Selection Methods: Different methods for selecting parent creatures, such as Roulette Wheel Selection and Tournament Selection.
        """
        text_widget = tk.Text(frame, wrap=tk.WORD, yscrollcommand=scrollbar.set)
        text_widget.insert(tk.END, explanation_text)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar.config(command=text_widget.yview)
    else:
        show_explanation.explanation_window.lift()  # Bring the window to the front if already open


# Create main application window
root = tk.Tk()
root.title("Evolution Simulator")

# Create a frame to hold the input fields and labels
input_frame = ttk.Frame(root, padding="10")
input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))

# Explanation button
explanation_button = ttk.Button(input_frame, text="Explanation", command=show_explanation)
explanation_button.grid(row=0, column=0, sticky=tk.W)

# Button to calculate overall statistics
calculate_statistics_button = ttk.Button(input_frame, text="Calculate Overall Statistics", command=calculate_overall_statistics)
calculate_statistics_button.grid(row=0, column=1, sticky=tk.W)

# Label to display overall statistics
overall_statistics_label = ttk.Label(input_frame, text="")
overall_statistics_label.grid(row=1, column=0, columnspan=2, sticky=tk.W)

# Population Size
population_size_label = ttk.Label(input_frame, text="Population Size:")
population_size_label.grid(row=2, column=0, sticky=tk.W)
population_size_entry = ttk.Entry(input_frame)
population_size_entry.insert(0, str(DEFAULT_POPULATION_SIZE))
population_size_entry.grid(row=2, column=1, sticky=tk.W)

# Gene Length
gene_length_label = ttk.Label(input_frame, text="Gene Length:")  # Add input field for gene length
gene_length_label.grid(row=3, column=0, sticky=tk.W)
gene_length_entry = ttk.Entry(input_frame)
gene_length_entry.insert(0, str(DEFAULT_GENE_LENGTH))
gene_length_entry.grid(row=3, column=1, sticky=tk.W)

# Mutation Rate
mutation_rate_label = ttk.Label(input_frame, text="Mutation Rate:")
mutation_rate_label.grid(row=4, column=0, sticky=tk.W)
mutation_rate_entry = ttk.Entry(input_frame)
mutation_rate_entry.insert(0, str(DEFAULT_MUTATION_RATE))
mutation_rate_entry.grid(row=4, column=1, sticky=tk.W)

# Number of Generations
num_generations_label = ttk.Label(input_frame, text="Number of Generations:")
num_generations_label.grid(row=5, column=0, sticky=tk.W)
num_generations_entry = ttk.Entry(input_frame)
num_generations_entry.insert(0, str(DEFAULT_NUM_GENERATIONS))
num_generations_entry.grid(row=5, column=1, sticky=tk.W)

# Parent Selection Method
parent_selection_method_label = ttk.Label(input_frame, text="Parent Selection Method:")
parent_selection_method_label.grid(row=6, column=0, sticky=tk.W)
parent_selection_method_combobox = ttk.Combobox(input_frame, values=["Roulette Wheel Selection", "Tournament Selection"])
parent_selection_method_combobox.current(0)
parent_selection_method_combobox.grid(row=6, column=1, sticky=tk.W)

# Start button
start_button = ttk.Button(input_frame, text="Start Evolution", command=start_evolution)
start_button.grid(row=7, column=0, columnspan=2, pady=5)

root.mainloop()
