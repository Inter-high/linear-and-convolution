"""
Dataset preparation module for MNIST experiments.

This module provides functions to load and preprocess the MNIST dataset 
with various transformations, including standard, half-rotated, and fully 
rotated versions. It also includes dataset splitting and DataLoader utilities.

Author: yumemonzo@gmail.com
Date: 2025-02-13
"""

import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Tuple, List


class CustomMNIST(Dataset):
    """
    Custom MNIST dataset where half of the images are transformed with rotation.
    """
    def __init__(self, root: str, train: bool, transform: transforms.Compose, 
                 rotation_transform: transforms.Compose, download: bool):
        self.dataset = MNIST(root, train=train, download=download)
        self.transform = transform
        self.rotation_transform = rotation_transform

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns a single image and its corresponding label.
        The first half of the dataset is kept as the original, 
        while the second half undergoes rotation transformation.
        """
        img, label = self.dataset[idx]
        if idx < len(self.dataset) // 2:
            return self.transform(img), label  # Original image
        else:
            return self.rotation_transform(img), label  # Rotated image


def get_transforms(flatten: bool = False) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns transformation pipelines for original and rotated images.
    
    Args:
        flatten (bool): Whether to flatten the image into a 1D tensor.

    Returns:
        Tuple of transformations (original transform, rotation transform).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        *( [transforms.Lambda(lambda x: x.view(-1))] if flatten else [] )
    ])

    rotation_transform = transforms.Compose([
        transforms.RandomRotation(degrees=(0, 360)),
        transforms.ToTensor(),
        *( [transforms.Lambda(lambda x: x.view(-1))] if flatten else [] )
    ])

    return transform, rotation_transform


def get_datasets(data_dir: str, flatten: bool, train: bool = True) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Creates and returns three variations of the MNIST dataset:
    - Standard MNIST dataset
    - Half-Rotation MNIST dataset (half original, half rotated)
    - Fully Rotated MNIST dataset (all images rotated)

    Args:
        data_dir (str): Path to the dataset directory.
        flatten (bool): Whether to flatten the images into 1D tensors.
        train (bool): Whether to load the training dataset.

    Returns:
        Tuple containing standard, half-rotated, and fully rotated datasets.
    """
    transform, rotation_transform = get_transforms(flatten)

    dataset = MNIST(root=data_dir, train=train, transform=transform, download=True)
    dataset_half_rotation = CustomMNIST(root=data_dir, train=train, transform=transform, 
                                        rotation_transform=rotation_transform, download=True)
    dataset_rotation = MNIST(root=data_dir, train=train, transform=rotation_transform, download=True)

    return dataset, dataset_half_rotation, dataset_rotation


def split_dataset(seed: int, dataset: Dataset, ratio: List[float] = [0.8, 0.2]) -> Tuple[Dataset, Dataset]:
    """
    Splits the dataset into training and validation sets.

    Args:
        seed (int): Random seed for reproducibility.
        dataset (Dataset): The dataset to be split.
        ratio (List[float]): The ratio for train/validation split.

    Returns:
        Tuple containing training and validation datasets.
    """
    generator = torch.Generator().manual_seed(seed)
    train_dataset, valid_dataset = random_split(dataset, ratio, generator)

    return train_dataset, valid_dataset


def get_train_loader(train_dataset: Dataset, valid_dataset: Dataset, 
                     batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader]:
    """
    Creates DataLoader instances for training and validation.

    Args:
        train_dataset (Dataset): The training dataset.
        valid_dataset (Dataset): The validation dataset.
        batch_size (int): Batch size for data loading.
        num_workers (int): Number of worker threads for data loading.

    Returns:
        Tuple containing training and validation DataLoaders.
    """
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size, shuffle=False, num_workers=num_workers)

    return train_dataloader, valid_dataloader


def get_test_loader(test_dataset: Dataset, batch_size: int, num_workers: int) -> DataLoader:
    """
    Creates a DataLoader instance for testing.

    Args:
        test_dataset (Dataset): The test dataset.
        batch_size (int): Batch size for data loading.
        num_workers (int): Number of worker threads for data loading.

    Returns:
        DataLoader for test data.
    """
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=num_workers)

    return test_dataloader
