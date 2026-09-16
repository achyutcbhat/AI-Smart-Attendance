import os
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from src.liveness.anti_spoof.model import AntiSpoofCNN


DATASET_PATH = "data/liveness"
MODEL_PATH = "models/anti_spoof/anti_spoof_cnn.pth"


def train():

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Load dataset
    dataset = datasets.ImageFolder(
        DATASET_PATH,
        transform=transform
    )

    print("\n===== DATASET =====")
    print("Classes:", dataset.classes)
    print("Total images:", len(dataset))

    # Show class distribution
    for class_name, class_index in dataset.class_to_idx.items():
        count = sum(
            1 for _, label in dataset.samples
            if label == class_index
        )
        print(f"{class_name}: {count}")

    # Train / validation split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False
    )

    # Model
    model = AntiSpoofCNN()

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    epochs = 20

    best_val_accuracy = 0.0

    print("\n===== TRAINING =====")

    for epoch in range(epochs):

        # -------------------------
        # Training
        # -------------------------
        model.train()

        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        train_accuracy = correct / total

        # -------------------------
        # Validation
        # -------------------------
        model.eval()

        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                outputs = model(images)

                predictions = torch.argmax(outputs, dim=1)

                val_correct += (
                    predictions == labels
                ).sum().item()

                val_total += labels.size(0)

        val_accuracy = val_correct / val_total

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- Loss: {total_loss:.4f} "
            f"- Train Acc: {train_accuracy:.2%} "
            f"- Val Acc: {val_accuracy:.2%}"
        )

        # Save best model
        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            os.makedirs(
                os.path.dirname(MODEL_PATH),
                exist_ok=True
            )

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )

    print("\n===== TRAINING COMPLETE =====")
    print(f"Best validation accuracy: {best_val_accuracy:.2%}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train()