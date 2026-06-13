import os
import kagglehub
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_data_loaders(batch_size=32, train_split=0.8):
    """
    Downloads the Real vs Fake dataset and returns PyTorch DataLoaders.
    """
    print("Downloading dataset from Kaggle...")
    # Download latest version of the dataset
    dataset_path = kagglehub.dataset_download("troykueh/real-vs-fake-faces-stylegan3")
    print(f"Dataset downloaded to: {dataset_path}")
    
    # Define standard image transformations (Resize, Normalize)
    transform = transforms.Compose([
        transforms.Resize((128, 128)), # Resize to save memory/processing power
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load entire dataset using ImageFolder (assumes subfolders represent classes)
    full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    
    # Calculate lengths for train/validation split
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, full_dataset.classes

if __name__ == "__main__":
    # Test file execution
    train_loader, val_loader, classes = get_data_loaders()
    print(f"Classes found: {classes}")
    print(f"Training batches: {len(train_loader)}")