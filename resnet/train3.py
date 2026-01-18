import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuration
class Config:
    # Paths
    DATA_DIR = 'apple_dataset/raw'
    OUTPUT_DIR = 'resnet'
    CHECKPOINTS_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
    RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
    LOGS_DIR = os.path.join(OUTPUT_DIR, 'logs')
    
    # Device - AUTO DETECT
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model parameters - AUTO ADJUST based on device
    IMG_SIZE = 224
    BATCH_SIZE = 32 if torch.cuda.is_available() else 8
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    NUM_WORKERS = 2
    
    # Classes
    CLASSES = ['alternaria', 'healthy', 'rust', 'scab']
    NUM_CLASSES = len(CLASSES)
    
    # Training parameters
    VALIDATION_SPLIT = 0.2
    RANDOM_SEED = 42
    EARLY_STOPPING_PATIENCE = 10

# Create directories
for directory in [Config.CHECKPOINTS_DIR, Config.RESULTS_DIR, Config.LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Custom Dataset
class AppleDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Data loading function
def load_dataset():
    print("Loading dataset...")
    image_paths = []
    labels = []
    class_counts = {}
    
    for class_idx, class_name in enumerate(Config.CLASSES):
        class_path = os.path.join(Config.DATA_DIR, class_name)
        if not os.path.exists(class_path):
            print(f"Warning: {class_path} does not exist!")
            continue
        
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        class_counts[class_name] = len(images)
        
        for img_name in images:
            image_paths.append(os.path.join(class_path, img_name))
            labels.append(class_idx)
    
    print("\nDataset Summary:")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count} images")
    print(f"Total images: {len(image_paths)}\n")
    
    return image_paths, labels

# Data transforms
def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

# Model creation
def create_model():
    print("Creating ResNet50 model...")
    model = models.resnet50(weights='DEFAULT')
    
    # Modify final layer for our classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, Config.NUM_CLASSES)
    
    return model.to(Config.DEVICE)

# Training function
def train_epoch(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{Config.NUM_EPOCHS} [Train]')
    
    for images, labels in pbar:
        images = images.to(Config.DEVICE)
        labels = labels.to(Config.DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{running_loss/len(train_loader):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

# Validation function
def validate(model, val_loader, criterion, epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{Config.NUM_EPOCHS} [Val]')
    
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({
                'loss': f'{running_loss/len(val_loader):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels

# Plot training history
def plot_history(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(history['val_loss'], label='Val Loss', marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Acc', marker='o')
    ax2.plot(history['val_acc'], label='Val Acc', marker='o')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to {save_path}")

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=Config.CLASSES,
                yticklabels=Config.CLASSES)
    plt.title('Confusion Matrix - ResNet50')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

# Main training function
def main():
    print("="*70)
    print("Apple Disease Classification - ResNet50 Training")
    print("="*70)
    print(f"\nDevice: {Config.DEVICE}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Image Size: {Config.IMG_SIZE}x{Config.IMG_SIZE}")
    print(f"Epochs: {Config.NUM_EPOCHS}")
    print(f"Learning Rate: {Config.LEARNING_RATE}\n")
    
    # Set random seeds
    torch.manual_seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)
    
    # Load dataset
    image_paths, labels = load_dataset()
    
    # Split dataset
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, 
        test_size=Config.VALIDATION_SPLIT,
        random_state=Config.RANDOM_SEED,
        stratify=labels
    )
    
    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}\n")
    
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Create datasets
    train_dataset = AppleDataset(train_paths, train_labels, train_transform)
    val_dataset = AppleDataset(val_paths, val_labels, val_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Create model
    model = create_model()
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    
    print("Starting training...\n")
    start_time = datetime.now()
    
    # Training loop
    for epoch in range(Config.NUM_EPOCHS):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, epoch)
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, epoch)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS} Summary:")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print("-" * 70)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            checkpoint_path = os.path.join(Config.CHECKPOINTS_DIR, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"✓ Best model saved! Val Acc: {val_acc:.2f}%\n")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Training complete
    end_time = datetime.now()
    training_time = end_time - start_time
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Training time: {training_time}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Save final model
    final_model_path = os.path.join(Config.CHECKPOINTS_DIR, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    
    # Plot training history
    history_plot_path = os.path.join(Config.RESULTS_DIR, 'training_history.png')
    plot_history(history, history_plot_path)
    
    # Load best model for final evaluation
    checkpoint = torch.load(os.path.join(Config.CHECKPOINTS_DIR, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final validation
    val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, Config.NUM_EPOCHS-1)
    
    # Classification report
    report = classification_report(val_labels, val_preds, 
                                  target_names=Config.CLASSES,
                                  digits=4)
    print("\nClassification Report:")
    print(report)
    
    # Save classification report
    report_path = os.path.join(Config.RESULTS_DIR, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    
    # Plot confusion matrix
    cm_path = os.path.join(Config.RESULTS_DIR, 'confusion_matrix.png')
    plot_confusion_matrix(val_labels, val_preds, cm_path)
    
    # Save training configuration and results
    results = {
        'model': 'ResNet50',
        'dataset': 'Apple Disease Classification',
        'classes': Config.CLASSES,
        'total_images': len(image_paths),
        'train_images': len(train_paths),
        'val_images': len(val_paths),
        'batch_size': Config.BATCH_SIZE,
        'epochs_trained': len(history['train_loss']),
        'best_val_accuracy': float(best_val_acc),
        'final_val_accuracy': float(val_acc),
        'training_time': str(training_time),
        'device': str(Config.DEVICE),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    results_path = os.path.join(Config.RESULTS_DIR, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nAll results saved to: {Config.OUTPUT_DIR}")
    print("\nFiles created:")
    print(f"  - {checkpoint_path}")
    print(f"  - {final_model_path}")
    print(f"  - {history_plot_path}")
    print(f"  - {cm_path}")
    print(f"  - {report_path}")
    print(f"  - {results_path}")

if __name__ == '__main__':
    main()
