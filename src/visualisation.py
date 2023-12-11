# visualisation.py

import matplotlib.pyplot as plt


def plot_training_loss(iteration, losses):
    if losses is not None:
        plt.plot(
            range(1, len(losses) + 1),
            losses,
            label=f"Iteration {iteration + 1}")
        plt.xlabel("Epochs")
        plt.ylabel("Training loss")
        plt.legend()
        plt.show()
    else:
        print("No training losses to plot.")
