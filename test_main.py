import random
from main import roulette_wheel_selection
from main import tournament_selection

# Test case 1: Empty population
population = []
fitness_values = []
result = roulette_wheel_selection(population, fitness_values)
assert result == (None, None), f"Expected: (None, None), Got: {result}"

# Test case 2: Single individual in population
population = ["Individual 1"]
fitness_values = [10]
result = roulette_wheel_selection(population, fitness_values)
assert result == ("Individual 1", "Individual 1"), f"Expected: ('Individual 1', 'Individual 1'), Got: {result}"

# Test case 3: Equal fitness values
population = ["Individual 1", "Individual 2", "Individual 3"]
fitness_values = [1, 1, 1]
random.seed(42)  # Set seed for reproducibility
result = roulette_wheel_selection(population, fitness_values)
assert result == ("Individual 2", "Individual 2"), f"Expected: ('Individual 2', 'Individual 2'), Got: {result}"

# Test case 4: Different fitness values
population = ["Individual 1", "Individual 2", "Individual 3"]
fitness_values = [1, 2, 3]
random.seed(42)  # Set seed for reproducibility
result = roulette_wheel_selection(population, fitness_values)
assert result == ("Individual 3", "Individual 3"), f"Expected: ('Individual 3', 'Individual 3'), Got: {result}"

print("All test cases passed!")


# Test case 1: Empty population
population = []
fitness_values = []
result = tournament_selection(population, fitness_values)
assert result == (None, None), f"Expected: (None, None), Got: {result}"

# Test case 2: Single individual in population
population = ["Individual 1"]
fitness_values = [10]
result = tournament_selection(population, fitness_values)
assert result == ("Individual 1", "Individual 1"), f"Expected: ('Individual 1', 'Individual 1'), Got: {result}"

# Test case 3: Equal fitness values
population = ["Individual 1", "Individual 2", "Individual 3"]
fitness_values = [1, 1, 1]
result = tournament_selection(population, fitness_values)
assert result == ("Individual 3", "Individual 3"), f"Expected: ('Individual 3', 'Individual 3'), Got: {result}"

# Test case 4: Different fitness values
population = ["Individual 1", "Individual 2", "Individual 3"]
fitness_values = [1, 2, 3]
result = tournament_selection(population, fitness_values)
assert result == ("Individual 3", "Individual 2"), f"Expected: ('Individual 3', 'Individual 2'), Got: {result}"

# Test case 5: Population size less than tournament size
population = ["Individual 1", "Individual 2"]
fitness_values = [1, 2]
result = tournament_selection(population, fitness_values)
assert result == ("Individual 2", "Individual 2"), f"Expected: ('Individual 2', 'Individual 2'), Got: {result}"

print("All test cases passed!")