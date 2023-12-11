# main.py

from src.quantum_data import generate_quantum_random_number, prepare_quantum_data

if __name__ == '__main__':

    # Generate and split data
    train_set, test_set = prepare_quantum_data(num_values=1000)

    # Model variables
    input_size = 1
    output_size = 1

    num_iterations = 5

    # Iterate through subsets of data
    for iteration in range(num_iterations):
        batch_size = 10

        # Indexes to splice training set
        start_index = iteration * batch_size
        end_index = min(start_index + batch_size, len(train_set))

        # Sample training
        sample = train_set[start_index:end_index]

        # Print sample
        print(f"Iteration {iteration + 1}")
        print(f"Sample values: {sample}")
