import os
import cv2
import pywt
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import timm

from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "dataset2"
BATCH_SIZE = 8
EPOCHS = 12
LR = 1e-4
IMG_SIZE = 299

# =========================
# PRNU / noise residual
# =========================
def extract_prnu_from_pil(pil_img):
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    coeffs2 = pywt.dwt2(gray, "haar")
    LL, (LH, HL, HH) = coeffs2
    prnu = LH + HL + HH

    prnu = (prnu - np.mean(prnu)) / (np.std(prnu) + 1e-8)

    prnu = cv2.resize(prnu, (IMG_SIZE, IMG_SIZE))
    prnu = np.expand_dims(prnu, axis=0)
    return torch.tensor(prnu, dtype=torch.float32)

# =========================
# Dataset
# =========================
class DualInputDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    self.samples.append((
                        os.path.join(cls_dir, fname),
                        self.class_to_idx[cls_name]
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        pil_img = Image.open(img_path).convert("RGB")

        rgb_tensor = self.transform(pil_img) if self.transform else pil_img
        prnu_tensor = extract_prnu_from_pil(pil_img)

        return rgb_tensor, prnu_tensor, label

# =========================
# Transform
# =========================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(299, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

full_dataset_for_labels = DualInputDataset(DATA_DIR, transform=val_transform)
print("classes:", full_dataset_for_labels.classes)
print("class_to_idx:", full_dataset_for_labels.class_to_idx)

indices = list(range(len(full_dataset_for_labels)))
labels = [full_dataset_for_labels.samples[i][1] for i in indices]

train_idx, val_idx = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

train_dataset_full = DualInputDataset(DATA_DIR, transform=train_transform)
val_dataset_full = DualInputDataset(DATA_DIR, transform=val_transform)

train_dataset = Subset(train_dataset_full, train_idx)
val_dataset = Subset(val_dataset_full, val_idx)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# =========================
# Model
# =========================
class XceptionBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("xception", pretrained=True, num_classes=0, global_pool="avg")

    def forward(self, x):
        return self.backbone(x)  # [B, 2048]

class PRNUBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        x = self.net(x)
        return x.view(x.size(0), -1)  # [B, 64]

class DualStreamModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.rgb_branch = XceptionBranch()
        self.prnu_branch = PRNUBranch()

        self.classifier = nn.Sequential(
            nn.Linear(2048 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, rgb, prnu):
        f1 = self.rgb_branch(rgb)
        f2 = self.prnu_branch(prnu)
        fused = torch.cat([f1, f2], dim=1)
        return self.classifier(fused)

model = DualStreamModel(num_classes=2).to(device)

#for param in model.rgb_branch.backbone.parameters():
    #param.requires_grad = False

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

best_val_acc = 0.0
patience = 3
no_improve = 0

# =========================
# Train
# =========================
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for rgb, prnu, labels in train_loader:
        rgb = rgb.to(device)
        prnu = prnu.to(device)
        labels = labels.to(device)

        outputs = model(rgb, prnu)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = outputs.argmax(dim=1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    train_acc = train_correct / train_total

    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for rgb, prnu, labels in val_loader:
            rgb = rgb.to(device)
            prnu = prnu.to(device)
            labels = labels.to(device)

            outputs = model(rgb, prnu)
            preds = outputs.argmax(dim=1)

            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "model.pth")
        print("✅ 已儲存最佳模型 model.pth")
        no_improve = 0
    else:
        no_improve += 1

    if no_improve >= patience:
        print("⛔ Early stopping")
        break

print(f"最佳驗證準確率: {best_val_acc:.4f}")