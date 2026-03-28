import torch
import pandas as pd
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from model import GalaxyCNN
from src.dataset import GalaxyDataset

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_df = pd.read_csv('../data/test.csv')
    test_df['label'] = test_df['label'].astype(int)

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0448, 0.0396, 0.0295],
                             std=[0.0878, 0.0729, 0.0646])
    ])

    test_dataset = GalaxyDataset(test_df, '../data/images/train/images_resized/', transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    model = GalaxyCNN().to(device)
    model.load_state_dict(torch.load('../best_model.pth', map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ── Accuracy ──────────────────────────────────────────────────────────────
    test_acc = (all_preds == all_labels).mean()
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Test Samples:  {len(all_labels)}")

    # ── Classification Report ─────────────────────────────────────────────────
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds,
                                target_names=['Smooth', 'Featured']))

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im)

    classes = ['Smooth', 'Featured']
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    # Add numbers inside the boxes
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')

    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('Confusion Matrix — Test Set')
    plt.tight_layout()
    plt.savefig('assets/confusion_matrix.png', dpi=150)
    plt.show()
    print("\nConfusion matrix saved to assets/confusion_matrix.png")