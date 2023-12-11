# quantum_ml_model.py

import torch
import torch.nn as nn


class QuantumModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(QuantumModel, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)


def train_model(model, inputs, targets, criterion, optimizer, num_epochs=10):
    losses = []

    inputs = torch.FloatTensor(inputs).view(-1, 1)
    targets = torch.FloatTensor(targets).view(-1, 1)

    for epoch in range(num_epochs):
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")

    print("Training finished.\n")

    return losses
