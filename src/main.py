# main.py

from src.quantum_random import generate_quantum_random_number

if __name__ == '__main__':
    num_iterations = 5

    for iteration in range(num_iterations):
        quantum_random_numbers = generate_quantum_random_number()

        print(f"Iteration {iteration + 1}: Quantum-generated random numbers: {quantum_random_numbers}")
