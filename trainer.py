"""
Trainer class for model training, validation, and testing.
Handles loss computation, accuracy evaluation, and model checkpointing.

Author: yumemonzo@gmail.com
Date: 2025-02-13
"""

import torch
import numpy as np


class Trainer:
    """Trainer class to handle model training, validation, and testing."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        device: str,
        logger,
    ) -> None:
        """
        Initialize the Trainer.

        Args:
            model (torch.nn.Module): The model to train.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            criterion (torch.nn.Module): Loss function.
            device (str): Device for training (CPU or GPU).
            logger: Logger for logging training progress.
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.logger = logger
        self.lowest_loss = float("inf")

    def train(self, train_dataloader: torch.utils.data.DataLoader) -> float:
        """
        Train the model for one epoch.

        Args:
            train_dataloader (torch.utils.data.DataLoader): Training data loader.

        Returns:
            float: Average training loss.
        """
        self.model.train()
        total_loss = 0.0

        for x, y in train_dataloader:
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            y_hat = self.model(x)

            loss = self.criterion(y_hat, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_dataloader)

    def valid(self, valid_dataloader: torch.utils.data.DataLoader) -> float:
        """
        Validate the model on the validation dataset.

        Args:
            valid_dataloader (torch.utils.data.DataLoader): Validation data loader.

        Returns:
            float: Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for x, y in valid_dataloader:
                x, y = x.to(self.device), y.to(self.device)

                y_hat = self.model(x)
                total_loss += self.criterion(y_hat, y).item()

        return total_loss / len(valid_dataloader)

    def test(
        self, test_dataloader: torch.utils.data.DataLoader
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """
        Evaluate the model on the test dataset.

        Args:
            test_dataloader (torch.utils.data.DataLoader): Test data loader.

        Returns:
            tuple: 
                - float: Overall test accuracy.
                - np.ndarray: Per-class error rates (percentage).
                - np.ndarray: Confusion matrix.
        """
        self.model.eval()
        total_correct = 0
        num_samples = 0
        num_classes = 10  # Assuming MNIST dataset with 10 classes
        class_total = torch.zeros(num_classes, dtype=torch.int32)
        class_wrong = torch.zeros(num_classes, dtype=torch.int32)

        # Initialize confusion matrix
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

        with torch.no_grad():
            for x, y in test_dataloader:
                num_samples += y.size(0)
                x = x.to(self.device)

                output = self.model(x)
                predictions = output.argmax(dim=1).cpu()

                total_correct += (predictions == y).sum().item()

                # Update per-class error and confusion matrix
                for true_label, pred_label in zip(y.numpy(), predictions.numpy()):
                    class_total[true_label] += 1
                    if true_label != pred_label:
                        class_wrong[true_label] += 1
                    confusion_matrix[true_label, pred_label] += 1

        accuracy = total_correct / num_samples
        class_error_rate = (class_wrong / class_total).numpy() * 100  # Convert to percentage

        return accuracy, class_error_rate, confusion_matrix

    def training(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        valid_dataloader: torch.utils.data.DataLoader,
        epochs: int,
        valid_interval: int,
        weight_path: str,
    ) -> tuple[list[float], list[float]]:
        """
        Train the model and validate at specified intervals.

        Args:
            train_dataloader (torch.utils.data.DataLoader): Training data loader.
            valid_dataloader (torch.utils.data.DataLoader): Validation data loader.
            epochs (int): Number of training epochs.
            valid_interval (int): Number of times validation should run.
            weight_path (str): Path to save the best model checkpoint.

        Returns:
            tuple:
                - list[float]: Training losses per epoch.
                - list[float]: Validation losses per validation interval.
        """
        train_losses = []
        valid_losses = []

        # Ensure validation runs only a limited number of times
        validation_interval = max(1, epochs // valid_interval)

        for epoch in range(epochs + 1):
            train_loss = self.train(train_dataloader)
            train_losses.append(train_loss)

            if epoch % validation_interval == 0:
                valid_loss = self.valid(valid_dataloader)
                valid_losses.append(valid_loss)

                if valid_loss < self.lowest_loss:
                    self.lowest_loss = valid_loss
                    torch.save(self.model.state_dict(), weight_path)
                    self.logger.info(
                        f"New best model saved at epoch {epoch} with Valid Loss: {valid_loss:.4f}"
                    )

                self.logger.info(
                    f"Epoch: {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}"
                )

        return train_losses, valid_losses
