# main.py

from src.quantum_data import prepare_quantum_data
from src.quantum_ml_model import QuantumModel, train_model
import torch
import torch.nn as nn
import torch.optim as optim

if __name__ == '__main__':

    # Generate and split data
    train_set, test_set = prepare_quantum_data(num_values=1000)

    # Model variables
    input_size = 1
    output_size = 1

    model = QuantumModel(input_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_inputs = torch.FloatTensor(train_set).view(-1, 1)
    train_targets = torch.FloatTensor(train_set).view(-1, 1)

    # Iterate through subsets of data
    num_iterations = 5
    for iteration in range(num_iterations):
        batch_size = 10

        # Indexes to splice training set
        start_index = iteration * batch_size
        end_index = min(start_index + batch_size, len(train_set))

        # Sample training
        sample_inputs = train_inputs[start_index:end_index]
        sample_targets = train_targets[start_index:end_index]

        train_model(
            model,
            sample_inputs,
            sample_targets,
            criterion,
            optimizer,
            num_epochs=10)

        print(f"Iteration {iteration + 1}")
        print(f"Sample values: {sample_inputs.squeeze().tolist()}")

    test_inputs = torch.FloatTensor(test_set).view(-1, 1)
    test_targets = torch.FloatTensor(test_set).view(-1, 1)

    model.eval()
    with torch.no_grad():
        test_outputs = model(test_inputs)

    print(f"\nTest set predictions: {test_outputs.squeeze().tolist()}")
