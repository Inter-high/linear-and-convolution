"""
Utility functions for setting random seeds, plotting training progress, and visualizing model performance.

Author: yumemonzo@gmail.com
Date: 2025-02-13
"""

import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns


def seed_everything(seed: int = 42) -> None:
    """
    Set seed for reproducibility across different modules.

    Args:
        seed (int, optional): The seed value to use. Defaults to 42.
    """
    random.seed(seed)  # Python built-in random seed
    torch.manual_seed(seed)  # PyTorch CPU seed
    torch.cuda.manual_seed(seed)  # PyTorch GPU seed

    torch.backends.cudnn.deterministic = True  # Ensures deterministic execution
    torch.backends.cudnn.benchmark = False  # Disable if model structure is not fixed


def plot_losses(
    title: str, 
    train_losses: list[float], 
    valid_losses: list[float], 
    epochs: int, 
    valid_interval: int, 
    save_path: str
) -> None:
    """
    Plot training and validation loss over epochs.

    Args:
        title (str): Title of the plot.
        train_losses (list[float]): Training loss per epoch.
        valid_losses (list[float]): Validation loss at intervals.
        epochs (int): Number of epochs.
        valid_interval (int): Number of validation steps.
        save_path (str): File path to save the plot.
    """
    plt.plot(range(epochs + 1), train_losses, label="Train Loss")  # X-axis: epochs

    validation_interval = max(1, epochs // valid_interval)
    valid_epochs = list(range(0, epochs + 1, validation_interval))
    plt.plot(valid_epochs, valid_losses, label="Valid Loss")  # Match validation points to epochs

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(title)
    plt.savefig(save_path)
    plt.clf()


def plot_classwise_accuracy(
    title: str, 
    class_error_rate: np.ndarray, 
    class_error_rate_rotation: np.ndarray, 
    save_path: str
) -> None:
    """
    Plot per-class accuracy comparison between original and rotated test sets.

    Args:
        title (str): Title of the plot.
        class_error_rate (np.ndarray): Error rate per class (original).
        class_error_rate_rotation (np.ndarray): Error rate per class (rotated).
        save_path (str): File path to save the plot.
    """
    class_accuracy = 100 - np.array(class_error_rate)
    class_accuracy_rotation = 100 - np.array(class_error_rate_rotation)

    classes = np.arange(len(class_accuracy))
    bar_width = 0.4

    plt.figure(figsize=(10, 5))
    plt.bar(classes - bar_width / 2, class_accuracy, width=bar_width, label="Original")
    plt.bar(classes + bar_width / 2, class_accuracy_rotation, width=bar_width, label="Full Rotation")

    plt.xlabel("Class")
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.xticks(classes)
    plt.ylim(0, 100)
    plt.legend()

    # Annotate bars with accuracy values
    for i in range(len(classes)):
        plt.text(classes[i] - bar_width / 2, class_accuracy[i] + 1, f"{class_accuracy[i]:.2f}%", ha="center", fontsize=10)
        plt.text(classes[i] + bar_width / 2, class_accuracy_rotation[i] + 1, f"{class_accuracy_rotation[i]:.2f}%", ha="center", fontsize=10)

    plt.savefig(save_path)
    plt.clf()


def plot_confusion_matrices(
    title: str, 
    confusion_matrix: np.ndarray, 
    confusion_matrix_rotation: np.ndarray, 
    save_path: str
) -> None:
    """
    Plot confusion matrices for original and rotated test sets.

    Args:
        title (str): Title of the plot.
        confusion_matrix (np.ndarray): Confusion matrix for original test set.
        confusion_matrix_rotation (np.ndarray): Confusion matrix for rotated test set.
        save_path (str): File path to save the plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", ax=axes[0])
    axes[0].set_title("Original")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    sns.heatmap(confusion_matrix_rotation, annot=True, fmt="d", cmap="Oranges", ax=axes[1])
    axes[1].set_title("Full Rotation")
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.clf()
