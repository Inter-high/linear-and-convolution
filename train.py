"""
Train and evaluate models on different datasets, including original, half-rotated, and fully rotated versions.
This script initializes models, trains them, evaluates performance, and saves results.

Author: yumemonzo@gmail.com
Date: 2025-02-13
"""

import os
import pickle
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig
import numpy as np
from data import (
    get_datasets,
    split_dataset,
    get_train_loader,
    get_test_loader,
)
from model.linear import LinearModel
from model.convolution import CNNModel
from trainer import Trainer
from utils import (
    seed_everything,
    plot_losses,
    plot_classwise_accuracy,
    plot_confusion_matrices,
)


def train_and_plot(
    cfg: DictConfig,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    test_loader_rot: torch.utils.data.DataLoader,
    model_name: str,
    logger: logging.Logger,
) -> tuple:
    """Train a model, evaluate it, and save the results."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize model
    if cfg['model']['linear']:
        model = LinearModel(cfg['model']['input_size'], cfg['model']['hidden_layers'], cfg['model']['num_classes']).to(device)
    else:
        model = CNNModel(cfg['model']['num_classes']).to(device)
    
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), cfg['train']['lr'])

    weight_path = os.path.join(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
        f"best_model_{model_name}.pth"
    )
    
    trainer = Trainer(model, optimizer, criterion, device, logger)
    
    logger.info(f"========== {model_name} training start ==========")
    train_losses, valid_losses = trainer.training(train_loader, valid_loader, cfg['train']['epochs'], cfg['train']['valid_interval'], weight_path)

    plot_losses(model_name, train_losses, valid_losses, cfg['train']['epochs'], cfg['train']['valid_interval'],
                os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, f"{model_name}_losses.jpg"))
    logger.info(f"========== {model_name} training end ==========\n")

    logger.info(f"========== {model_name} predict start ==========")
    test_acc, class_err, conf_matrix = trainer.test(test_loader)
    test_rot_acc, class_err_rot, conf_matrix_rot = trainer.test(test_loader_rot)

    plot_classwise_accuracy(model_name, class_err, class_err_rot,
                            os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, f"{model_name}_class_wises_accuracy.jpg"))
    plot_confusion_matrices(model_name, conf_matrix, conf_matrix_rot,
                            os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, f"{model_name}_confusion_matrix.jpg"))
    
    logger.info(f"Test Accuracy - Original: {test_acc:.4f}, Full Rotation: {test_rot_acc:.4f}")
    logger.info(f"========== {model_name} predict end ==========\n")

    return train_losses, valid_losses, test_acc, test_rot_acc, class_err, class_err_rot, conf_matrix, conf_matrix_rot


@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    """Main function to train and evaluate models."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Set seed for reproducibility
    seed_everything(cfg['seed'])

    # Load and split datasets
    dataset, dataset_half_rot, dataset_full_rot = get_datasets(cfg['data']['data_dir'], cfg['data']['flatten'])
    test_dataset, _, test_dataset_full_rot = get_datasets(cfg['data']['data_dir'], cfg['data']['flatten'], train=False)

    train_set, valid_set = split_dataset(cfg['seed'], dataset)
    train_half_rot, valid_half_rot = split_dataset(cfg['seed'], dataset_half_rot)
    train_full_rot, valid_full_rot = split_dataset(cfg['seed'], dataset_full_rot)

    # Create data loaders
    batch_size, num_workers = cfg['data']['batch_size'], cfg['data']['num_workers']
    train_loader, valid_loader = get_train_loader(train_set, valid_set, batch_size, num_workers)
    train_loader_half_rot, valid_loader_half_rot = get_train_loader(train_half_rot, valid_half_rot, batch_size, num_workers)
    train_loader_full_rot, valid_loader_full_rot = get_train_loader(train_full_rot, valid_full_rot, batch_size, num_workers)

    test_loader = get_test_loader(test_dataset, batch_size, num_workers)
    test_loader_full_rot = get_test_loader(test_dataset_full_rot, batch_size, num_workers)

    # Train and evaluate models on different datasets
    results = {
        "Dataset": train_and_plot(cfg, train_loader, valid_loader, test_loader, test_loader_full_rot, "dataset", logger),
        "Half Rotation": train_and_plot(cfg, train_loader_half_rot, valid_loader_half_rot, test_loader, test_loader_full_rot, "dataset_half_rotation", logger),
        "Full Rotation": train_and_plot(cfg, train_loader_full_rot, valid_loader_full_rot, test_loader, test_loader_full_rot, "dataset_full_rotation", logger),
    }

    # Save results
    save_path = os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, "train_results.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(results, f)

    # Plot loss curves
    valid_interval = max(1, cfg['train']['epochs'] // cfg['train']['valid_interval'])
    valid_epochs = list(range(0, cfg['train']['epochs'] + 1, valid_interval))

    for name, (train_losses, valid_losses, *_) in results.items():
        plt.plot(train_losses, label=f"{name} Train Loss")
        plt.plot(valid_epochs, valid_losses, label=f"{name} Valid Loss")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("All Datasets Loss")
    plt.savefig(os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, "all_dataset_losses.jpg"))
    plt.clf()

    # Plot test accuracy comparison
    datasets = ["Dataset", "Half Rotation", "Full Rotation"]
    test_accs = [results[name][2] for name in datasets]
    test_rot_accs = [results[name][3] for name in datasets]

    x = np.arange(len(datasets))
    bar_width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - bar_width / 2, test_accs, bar_width, label="Test Accuracy (Original)")
    ax.bar(x + bar_width / 2, test_rot_accs, bar_width, label="Test Accuracy (Full Rotation)")

    ax.set_xlabel("Dataset Type")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Test Accuracy Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()

    plt.savefig(os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, "all_dataset_test_accuracy.jpg"))
    plt.clf()


if __name__ == "__main__":
    my_app()
