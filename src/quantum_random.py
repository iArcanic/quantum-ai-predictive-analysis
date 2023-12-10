# quantum_random.py

from qiskit import QuantumCircuit, Aer, transpile, assemble


def generate_quantum_random_number():
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.measure_all()

    simulator = Aer.get_backend('qasm_simulator')
    compiled_circuit = transpile(qc, simulator)
    result = simulator.run(assemble(compiled_circuit)).result()

    counts = result.get_counts()
    quantum_random_numbers = [int(key) for key in counts.keys()]

    return quantum_random_numbers
