import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- Model Definition (Hardcoded inside this script) ---
class PyTorchCNN(nn.Module):
    def __init__(self, num_classes):
        super(PyTorchCNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(128 * 16 * 16, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# --- Custom Dataset ---
class ProductDataset(Dataset):
    def __init__(self, dataframe, transform=None, root_dir=""):
        self.dataframe = dataframe
        self.transform = transform
        self.root_dir = root_dir
        self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(self.dataframe['product'].unique()))}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = os.path.join(self.root_dir, row['image_path'])
        try:
            img = Image.open(img_path).convert('RGB')
            label = self.class_to_idx[row['product']]
            if self.transform:
                img = self.transform(img)
            return img, label
        except FileNotFoundError:
            print(f"Warning: Image not found at {img_path}. Returning next item.")
            return self.__getitem__((idx + 1) % len(self))


# --- Main Training Function ---
def main():
    # Configuration
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_CSV = os.path.join(PROJECT_ROOT, "data", "CNN_Model_Train_Data_FIXED.csv")
    MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "models", "pytorch_product_cnn.pth")
    BATCH_SIZE = 32
    EPOCHS = 50
    IMAGE_SIZE = (128, 128)
    LEARNING_RATE = 0.001

    # Data transformations
    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load and prepare data
    if not os.path.exists(DATA_CSV):
        raise FileNotFoundError(f"Data file not found: {DATA_CSV}. Please run data preparation scripts.")
    df = pd.read_csv(DATA_CSV).dropna(subset=['image_path', 'product'])

    min_samples = 5
    class_counts = df['product'].value_counts()
    valid_classes = class_counts[class_counts >= min_samples].index
    df_filtered = df[df['product'].isin(valid_classes)]
    print(f"Original dataset size: {len(df)}. Filtered dataset size: {len(df_filtered)}.")

    # Split data
    train_df, val_df = train_test_split(df_filtered, test_size=0.2, stratify=df_filtered['product'], random_state=42)

    # Create datasets
    train_dataset = ProductDataset(train_df, train_transform, PROJECT_ROOT)
    val_dataset = ProductDataset(val_df, val_transform, PROJECT_ROOT)
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty. Check data paths and filtering.")

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model, loss, and optimizer
    num_classes = len(train_dataset.class_to_idx)
    model = PyTorchCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)

    # Training loop
    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        # Validation
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        val_epoch_loss = val_loss / len(val_dataset)
        val_epoch_acc = val_corrects.double() / len(val_dataset)
        print(
            f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc:.4f}")
        scheduler.step(val_epoch_loss)

        # Save best model
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_to_idx': train_dataset.class_to_idx,
                'idx_to_class': train_dataset.idx_to_class,
            }, MODEL_SAVE_PATH)
            print(f"✅ Best model saved to {MODEL_SAVE_PATH} with validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
