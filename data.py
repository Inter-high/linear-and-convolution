import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split


class CustomMNIST(Dataset):
    def __init__(self, root, train, transform, rotation_transform, download):
        self.dataset = MNIST(root, train=train, download=download)
        self.transform = transform
        self.rotation_transform = rotation_transform
    
    def __len__(self): 
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if idx < len(self.dataset) // 2:
            return self.transform(img), label  # 원본
        else:
            return self.rotation_transform(img), label  # 회전 적용
        

def get_transforms(flatten=False):
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


def get_datasets(data_dir, flatten, train=True):
    transform, rotation_transform = get_transforms(flatten)

    dataset = MNIST(root=data_dir, train=train, transform=transform, download=True)
    dataset_half_rotation = CustomMNIST(root=data_dir, train=train, transform=transform, rotation_transform=rotation_transform, download=True)
    dataset_rotation = MNIST(root=data_dir, train=train, transform=rotation_transform, download=True)

    return dataset, dataset_half_rotation, dataset_rotation


def split_dataset(seed, dataset, ratio=[0.8, 0.2]):
    generator = torch.Generator().manual_seed(seed)
    train_dataset, valid_dataset = random_split(dataset, ratio, generator)

    return train_dataset, valid_dataset


def get_train_loader(train_dataset, valid_dataset, batch_size, num_workers):
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size, shuffle=False, num_workers=num_workers)

    return train_dataloader, valid_dataloader

def get_test_loader(test_dataset, batch_size, num_workers):
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=num_workers)

    return test_dataloader
