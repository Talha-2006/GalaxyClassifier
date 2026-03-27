import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from dataset import GalaxyDataset
from model import GalaxyCNN
from engine import train_one_epoch, validate

if __name__ == '__main__':

    # Config
    IMG_DIR = 'data/images/train/images_resized/'
    TRAIN_CSV = 'data/train.csv'
    VAL_CSV = 'data/val.csv'
    NUM_EPOCHS = 30
    BATCH_SIZE = 64
    LR = 3e-4
    PATIENCE = 5

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    train_df['label'] = train_df['label'].astype(int)
    val_df['label'] = val_df['label'].astype(int)

    # Transforms
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0448, 0.0396, 0.0295],
                             std=[0.0878, 0.0729, 0.0646])
    ])

    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0448, 0.0396, 0.0295],
                             std=[0.0878, 0.0729, 0.0646])
    ])

    # Datasets
    train_dataset = GalaxyDataset(train_df, IMG_DIR, transform=train_transforms)
    val_dataset = GalaxyDataset(val_df, IMG_DIR, transform=val_transforms)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)

    # Model
    model = GalaxyCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5)

    # Training loop
    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        scheduler.step(val_loss)
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print("Model saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Save learning curves
    torch.save({
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }, 'training_history.pt')

    print("\nTraining complete.")
    print(f"Best val loss: {best_val_loss:.4f}")