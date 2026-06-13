import torch
import torch.nn as nn
from torchvision import models

def build_model(num_classes=2):
    """
    Creates a MobileNetV3 model modified for Binary Classification.
    Works dynamically across older and newer versions of torchvision.
    """
    # Check if the modern Weights API is supported in the current torchvision version
    if hasattr(models, "MobileNetV3_Small_Weights"):
        # Modern way (torchvision >= 0.13)
        weights = models.MobileNetV3_Small_Weights.DEFAULT
        model = models.mobilenet_v3_small(weights=weights)
    else:
        # Legacy fallback way (torchvision < 0.13)
        model = models.mobilenet_v3_small(pretrained=True)
    
    # Freeze the backbone parameters so we don't destroy pre-trained features
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace the final classification head
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    
    return model

def train_model(model, train_loader, val_loader, epochs=1):
    """
    A simplified training loop designed to run quickly.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier[3].parameters(), lr=0.001)
    
    print(f"Starting training on device: {device}...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f}")
        
    # Save the trained model weights locally
    torch.save(model.state_dict(), "model_weights.pth")
    print("Model saved as model_weights.pth")
    return model

if __name__ == "__main__":
    # Test model building
    model = build_model()
    print(model)