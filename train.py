import os
import pickle
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import hydra
from data import get_datasets, split_dataset, get_train_loader, get_test_loader
from model.linear import LinearModel
from model.convolution import CNNModel
from trainer import Trainer
from utils import seed_everything, plot_losses, plot_classwise_accuracy, plot_confusion_matrices
from omegaconf import DictConfig
import numpy as np


def train_and_plot(cfg, train_dataloader, valid_dataloader, test_dataloader, test_dataloader_rotation, model_name, logger):
    """모델 학습 및 손실 그래프 저장"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 모델, 손실 함수, 옵티마이저 초기화
    if cfg['model']['linear']:
        model = LinearModel(cfg['model']['input_size'], cfg['model']['hidden_layers'], cfg['model']['num_classes']).to(device)
    else:
        model = CNNModel(cfg['model']['num_classes']).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), cfg['train']['lr'])
    
    weight_path = os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, f"best_model_{model_name}.pth")
    trainer = Trainer(model, optimizer, criterion, device, logger)

    logger.info(f"========== {model_name} training start ==========")
    train_losses, valid_losses = trainer.training(train_dataloader, valid_dataloader, cfg['train']['epochs'], cfg['train']['valid_interval'], weight_path)

    plot_losses(model_name, train_losses, valid_losses, cfg['train']['epochs'], cfg['train']['valid_interval'], os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, f"{model_name}_losses.jpg"))
    logger.info(f"========== {model_name} training end ==========\n")

    logger.info(f"========== {model_name} predict start ==========")
    test_acc, class_error_rate, confusion_matrix = trainer.test(test_dataloader)
    test_rotation_acc, class_error_rate_rotation, confusion_matrix_rotation = trainer.test(test_dataloader_rotation)
    plot_classwise_accuracy(model_name, class_error_rate, class_error_rate_rotation, os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, f"{model_name}_class_wises_accuracy.jpg"))
    plot_confusion_matrices(model_name, confusion_matrix, confusion_matrix_rotation, os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, f"{model_name}_confusion_matrix.jpg"))
    logger.info(f"Test Accuracy - Original: {test_acc:.4f}, Full Rotation: {test_rotation_acc:.4f}")
    logger.info(f"========== {model_name} predict end ==========\n")

    return train_losses, valid_losses, test_acc, test_rotation_acc, class_error_rate, class_error_rate_rotation, confusion_matrix, confusion_matrix_rotation


@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Seed 설정
    seed_everything(cfg['seed'])

    # 데이터셋 로드
    dataset, dataset_half_rotation, dataset_rotation = get_datasets(cfg['data']['data_dir'], cfg['data']['flatten'])
    test_dataset, _, test_dataset_rotation = get_datasets(cfg['data']['data_dir'], cfg['data']['flatten'], train=False)
    logger.info(f"Dataset sizes - Original: {len(dataset)}, Half Rotation: {len(dataset_half_rotation)}, Full Rotation: {len(dataset_rotation)}")
    logger.info(f"Test Dataset sizes - Original: {len(test_dataset)}, Full Rotation: {len(test_dataset_rotation)}")

    # 데이터셋 분할
    train_dataset, valid_dataset = split_dataset(cfg['seed'], dataset)
    train_dataset_half_rotation, valid_dataset_half_rotation = split_dataset(cfg['seed'], dataset_half_rotation)
    train_dataset_rotation, valid_dataset_rotation = split_dataset(cfg['seed'], dataset_rotation)

    # 데이터로더 생성
    batch_size, num_workers = cfg['data']['batch_size'], cfg['data']['num_workers']
    train_dataloader, valid_dataloader = get_train_loader(train_dataset, valid_dataset, batch_size, num_workers)
    train_dataloader_half_rotation, valid_dataloader_half_rotation = get_train_loader(train_dataset_half_rotation, valid_dataset_half_rotation, batch_size, num_workers)
    train_dataloader_rotation, valid_dataloader_rotation = get_train_loader(train_dataset_rotation, valid_dataset_rotation, batch_size, num_workers)
    test_dataloader = get_test_loader(test_dataset, cfg['data']['batch_size'], cfg['data']['num_workers'])
    test_dataloader_rotation = get_test_loader(test_dataset_rotation, cfg['data']['batch_size'], cfg['data']['num_workers'])

    logger.info(f"Dataloader sizes - Train: {len(train_dataloader)}, Valid: {len(valid_dataloader)}, Test: {len(test_dataloader)}")

    # 모델 학습 및 그래프 저장
    train_losses, valid_losses, test_acc, test_rotation_acc, class_error_rate, class_error_rate_rotation, confusion_matrix, confusion_matrix_rotation = train_and_plot(cfg, train_dataloader, valid_dataloader, test_dataloader, test_dataloader_rotation, "dataset", logger)
    train_losses_half_rotation, valid_losses_half_rotation, test_acc_half_rotation, test_rotation_acc_half_rotation, class_error_rate_half_rotation, class_error_rate_rotation_half_rotation, confusion_matrix_half_rotation, confusion_matrix_rotation_half_rotation = train_and_plot(cfg, train_dataloader_half_rotation, valid_dataloader_half_rotation, test_dataloader, test_dataloader_rotation, "dataset_half_rotation", logger)
    train_losses_rotation, valid_losses_rotation, test_acc_rotation, test_rotation_acc_rotation, class_error_rate_rotation, class_error_rate_rotation_rotation, confusion_matrix_rotation, confusion_matrix_rotation_rotation = train_and_plot(cfg, train_dataloader_rotation, valid_dataloader_rotation, test_dataloader, test_dataloader_rotation, "dataset_full_rotation", logger)

    # 값 저장
    data_dict = {
        "Dataset": {
            "train_losses": train_losses,
            "valid_losses": valid_losses,
            "test_acc": test_acc,
            "test_rotation_acc": test_rotation_acc,
            "class_error_rate": class_error_rate,
            "class_error_rate_rotation": class_error_rate_rotation,
            "confusion_matrix": confusion_matrix,
            "confusion_matrix_rotation": confusion_matrix_rotation,
        },
        "Half Rotation": {
            "train_losses": train_losses_half_rotation,
            "valid_losses": valid_losses_half_rotation,
            "test_acc": test_acc_half_rotation,
            "test_rotation_acc": test_rotation_acc_half_rotation,
            "class_error_rate": class_error_rate_half_rotation,
            "class_error_rate_rotation": class_error_rate_rotation_half_rotation,
            "confusion_matrix": confusion_matrix_half_rotation,
            "confusion_matrix_rotation": confusion_matrix_rotation_half_rotation,
        },
        "Full Rotation": {
            "train_losses": train_losses_rotation,
            "valid_losses": valid_losses_rotation,
            "test_acc": test_acc_rotation,
            "test_rotation_acc": test_rotation_acc_rotation,
            "class_error_rate": class_error_rate_rotation,
            "class_error_rate_rotation": class_error_rate_rotation_rotation,
            "confusion_matrix": confusion_matrix_rotation,
            "confusion_matrix_rotation": confusion_matrix_rotation_rotation,
        }
    }

    # pkl 파일로 저장
    with open(os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, "train_results.pkl"), "wb") as f:
        pickle.dump(data_dict, f)

    # 모든 데이터셋 학습 결과를 하나의 그래프에 저장
    validation_interval = max(1, cfg['train']['epochs'] // cfg['train']['valid_interval']) 
    valid_epochs = list(range(0, cfg['train']['epochs'] + 1, validation_interval))

    plt.plot(train_losses, label="Train Loss")
    plt.plot(valid_epochs, valid_losses, label="Valid Loss")
    plt.plot(train_losses_half_rotation, label="Half Rotation Train Loss")
    plt.plot(valid_epochs, valid_losses_half_rotation, label="Half Rotation Valid Loss")
    plt.plot(train_losses_rotation, label="Rotation Train Loss")
    plt.plot(valid_epochs, valid_losses_rotation, label="Rotation Valid Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("All datasets Loss")
    plt.savefig(os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, "all_dataset_losses.jpg"))
    plt.clf()

    # ✅ Test Accuracy 그래프 추가
    datasets = ["Dataset", "Half Rotation", "Full Rotation"]
    test_accuracies = [test_acc, test_acc_half_rotation, test_acc_rotation]
    test_rotation_accuracies = [test_rotation_acc, test_rotation_acc_half_rotation, test_rotation_acc_rotation]

    x = np.arange(len(datasets))  # X축 위치
    bar_width = 0.35  # 막대 너비

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - bar_width/2, test_accuracies, bar_width, label="Test Accuracy (Original)")
    rects2 = ax.bar(x + bar_width/2, test_rotation_accuracies, bar_width, label="Test Accuracy (Full Rotation)")

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
