# quantum_data.py

from qiskit import QuantumCircuit, Aer, transpile, assemble
import numpy as np


def generate_quantum_random_number(num_bits=1, num_values=1):
    random_numbers = []

    for i in range(num_values):

        # Create a quantum circuit with the specified number of qubits
        qc = QuantumCircuit(1)

        # Apply Hadamard gate to create a superposition
        qc.h(0)

        # Measure qubits
        qc.measure_all()

        # Run quantum circuit using Aer simulator
        simulator = Aer.get_backend('qasm_simulator')

        # Compile the circuit for runtime for the simulator
        compiled_circuit = transpile(qc, simulator)
        result = simulator.run(assemble(compiled_circuit)).result()

        # Extract the counts of each measurement outcome
        counts = result.get_counts()

        # Convert counts to a random number
        key = list(counts.keys())[0]
        random_num = int(key, 2)
        random_numbers.append(random_num)

    return random_numbers


def prepare_quantum_data(num_values=1000):

    # Generate 5000 raw quantum random numbers
    quantum_dataset = []
    for i in range(num_values):
        random_num = generate_quantum_random_number()
        quantum_dataset.append(random_num[0])

    # Data preprocessing
    max_value = max(quantum_dataset)

    # Normalise data to 0-1
    quantum_dataset = np.array(quantum_dataset) / max_value

    # Split data for training (80%) and testing (20%)
    split_index = int(len(quantum_dataset) * 0.8)
    train_set = quantum_dataset[:split_index]
    test_set = quantum_dataset[split_index:]

    return train_set, test_set
