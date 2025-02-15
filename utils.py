import torch
import seaborn as sns
import numpy as np
import random
import matplotlib.pyplot as plt


def seed_everything(seed=42):
    random.seed(seed)  # Python 기본 랜덤 시드 고정
    torch.manual_seed(seed)  # PyTorch CPU 시드 고정
    torch.cuda.manual_seed(seed)  # PyTorch GPU 시드 고정

    torch.backends.cudnn.deterministic = True  # 연산의 결정적(Deterministic) 실행 보장
    torch.backends.cudnn.benchmark = False  # 모델 구조가 고정되지 않으면 비활성화


def plot_losses(title, train_losses, valid_losses, epochs, valid_interval, save_path):
    plt.plot(range(epochs + 1), train_losses, label="Train Loss")  # X축: 전체 epoch

    validation_interval = max(1, epochs // valid_interval) 
    valid_epochs = list(range(0, epochs + 1, validation_interval))
    plt.plot(valid_epochs, valid_losses, label="Valid Loss")  # Validation이 실행된 epoch에 맞춰 X축 지정

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(title)
    plt.savefig(save_path)
    plt.clf()


def plot_classwise_accuracy(title, class_error_rate, class_error_rate_rotation, save_path):
    class_accuracy = 100 - np.array(class_error_rate)
    class_accuracy_rotation = 100 - np.array(class_error_rate_rotation)

    classes = np.arange(len(class_accuracy))
    bar_width = 0.4

    plt.figure(figsize=(10, 5))
    plt.bar(classes - bar_width/2, class_accuracy, width=bar_width, label="Original")
    plt.bar(classes + bar_width/2, class_accuracy_rotation, width=bar_width, label="Full Rotation")

    plt.xlabel("Class")
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.xticks(classes)
    plt.ylim(0, 100)
    plt.legend()

    for i in range(len(classes)):
        plt.text(classes[i] - bar_width/2, class_accuracy[i] + 1, f"{class_accuracy[i]:.2f}%", ha='center', fontsize=10)
        plt.text(classes[i] + bar_width/2, class_accuracy_rotation[i] + 1, f"{class_accuracy_rotation[i]:.2f}%", ha='center', fontsize=10)

    plt.savefig(save_path)
    plt.clf()


def plot_confusion_matrices(title, confusion_matrix, confusion_matrix_rotation, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=16, fontweight='bold')

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
