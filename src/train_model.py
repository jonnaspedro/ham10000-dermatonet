import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import joblib
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
EPOCHS = 30
IMG_SIZE = 224
LEARNING_RATE = 0.0001

CLASSES = {
    'akiec': 0,
    'bcc': 1,
    'bkl': 2,
    'df': 3,
    'mel': 4,
    'nv': 5,
    'vasc': 6
}

CLASS_NAMES = list(CLASSES.keys())

print(f"Dispositivo: {DEVICE}")
print(f"Classes: {CLASS_NAMES}")

path = "dataset"

metadata_path = os.path.join(path, "HAM10000_metadata.csv")
df = pd.read_csv(metadata_path)

print(f"\nTotal de imagens: {len(df)}")
print("Distribuição das classes:")
print(df['dx'].value_counts())

img_dir1 = os.path.join(path, "HAM10000_images_part_1")
img_dir2 = os.path.join(path, "HAM10000_images_part_2")

def get_image_path(image_id):
    """Encontra o caminho completo da imagem"""
    path1 = os.path.join(img_dir1, f"{image_id}.jpg")
    path2 = os.path.join(img_dir2, f"{image_id}.jpg")
    
    if os.path.exists(path1):
        return path1
    elif os.path.exists(path2):
        return path2
    else:
        return None

df['image_path'] = df['image_id'].apply(get_image_path)
df = df.dropna(subset=['image_path'])
df['label'] = df['dx'].map(CLASSES)

print(f"✅ Imagens encontradas: {len(df)}")

train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

print("\nSplit dos dados:")
print(f"   Treino: {len(train_df)}")
print(f"   Validação: {len(val_df)}")
print(f"   Teste: {len(test_df)}")

class HAM10000Dataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.df.loc[idx, 'image_path']
        image = Image.open(img_path).convert('RGB')
        label = self.df.loc[idx, 'label']
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = HAM10000Dataset(train_df, transform=train_transform)
val_dataset = HAM10000Dataset(val_df, transform=val_transform)
test_dataset = HAM10000Dataset(test_df, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

def create_model(num_classes=7):
    """Cria modelo ResNet50 pré-treinado"""
    model = models.resnet50(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
    
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    return model

model = create_model(num_classes=len(CLASSES))
model = model.to(DEVICE)

print("\nModelo: ResNet50 com Transfer Learning")

class_counts = df['dx'].value_counts()
class_weights = 1.0 / torch.tensor([class_counts[name] for name in CLASS_NAMES], dtype=torch.float)
class_weights = class_weights / class_weights.sum() * len(CLASSES)
class_weights = class_weights.to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(loader, desc="Treinando"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validando"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(loader), 100. * correct / total

print("\nIniciando treinamento...\n")

history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': []
}

best_val_acc = 0.0

for epoch in range(EPOCHS):
    print(f"\n{'='*60}")
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"{'='*60}")
    
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc = validate(model, val_loader, criterion)
    
    scheduler.step(val_loss)
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    print("\nResultados:")
    print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"   Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'generated/dermatonet_best.pth')
        print(f"   Melhor modelo salvo! (Val Acc: {val_acc:.2f}%)")

print("\n" + "="*60)
print("AVALIAÇÃO FINAL NO CONJUNTO DE TESTE")
print("="*60)

model.load_state_dict(torch.load('generated/dermatonet_best.pth'))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testando"):
        images = images.to(DEVICE)
        outputs = model(images)
        _, predicted = outputs.max(1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

print("\nRELATÓRIO DE CLASSIFICAÇÃO:")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Matriz de Confusão - DermatoNet')
plt.ylabel('Verdadeiro')
plt.xlabel('Predito')
plt.tight_layout()
plt.savefig('generated/confusion_matrix.png', dpi=300)
print("Matriz de confusão salva em: confusion_matrix.png")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].plot(history['train_loss'], label='Train Loss')
axes[0].plot(history['val_loss'], label='Val Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Loss ao longo do treinamento')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history['train_acc'], label='Train Acc')
axes[1].plot(history['val_acc'], label='Val Acc')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Acurácia ao longo do treinamento')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('generated/training_history.png', dpi=300)
print("Histórico de treinamento salvo em: training_history.png")

metadata = {
    'class_names': CLASS_NAMES,
    'img_size': IMG_SIZE,
    'best_val_acc': best_val_acc,
    'test_acc': 100. * sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
}

joblib.dump(metadata, 'generated/model_metadata.pkl')
print("Metadados salvos em: model_metadata.pkl")

print("\n" + "="*60)
print("TREINAMENTO CONCLUÍDO COM SUCESSO!")
print("="*60)
print(f"✅ Melhor acurácia de validação: {best_val_acc:.2f}%")
print(f"✅ Acurácia no teste: {metadata['test_acc']:.2f}%")
print("✅ Modelo salvo: dermatonet_best.pth")
print("✅ Pronto para deploy no Streamlit!")
print("="*60)