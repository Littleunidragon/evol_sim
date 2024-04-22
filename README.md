# [Evolution Simulator](https://github.com/Littleunidragon/evol_sim)

This is a Python-based evolutionary simulation program that uses neural networks to evolve creatures over multiple generations. The program allows users to specify various parameters such as population size, gene length, mutation rate, and number of generations to observe how creatures evolve over time.

## Features:

- Evolution of creatures using neural networks
- Adjustable parameters for population size, gene length, mutation rate, and number of generations
- Support for different parent selection methods including Roulette Wheel Selection and Tournament Selection
- Visualization of evolution process through fitness plots

## Getting Started:

1. Install Python if not already installed. You can download Python from the [official website](https://www.python.org/).
2. Clone or download the repository to your local machine.
3. Install the required dependencies using the provided `requirements.txt` file.
   ```bash
   pip install -r requirements.txt
   ```
4. Run the `main.py` file to start the simulation.
   ```bash
   python main.py
   ```

## Usage:

- Upon running the program, a GUI window will appear allowing you to specify the simulation parameters such as population size, gene length, mutation rate, number of generations, and parent selection method.
- Click on the "Start Evolution" button to begin the simulation.
- The program will display the evolution process in the console and plot the fitness of the best creature over generations.
- You can also calculate overall statistics by clicking the "Calculate Overall Statistics" button.
- For more detailed information about the simulation, you can click on the "Explanation" button to view the explanation window.

## License:

This project is licensed under the terms of the MIT License. See the [LICENSE](LICENSE) file for more information.