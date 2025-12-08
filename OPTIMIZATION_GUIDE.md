# 🚀 DermatoNet - Guia de Otimização e Boas Práticas

## 📊 Otimizações de Performance

### 1. Melhorar Acurácia do Modelo

#### A. Experimentar Outras Arquiteturas

```python
# EfficientNet-B4 (mais eficiente)
from torchvision import models
model = models.efficientnet_b4(pretrained=True)
model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.classifier[1].in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 7)
)

# DenseNet-161 (conexões densas)
model = models.densenet161(pretrained=True)
model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.classifier.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 7)
)

# Vision Transformer (ViT) - estado da arte
model = models.vit_b_16(pretrained=True)
model.heads = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.heads.head.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 7)
)
```

#### B. Técnicas de Ensemble

```python
# Ensemble de múltiplos modelos
class EnsembleModel(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = models
    
    def forward(self, x):
        predictions = []
        for model in self.models:
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        # Média das predições
        ensemble_pred = torch.stack(predictions).mean(dim=0)
        return ensemble_pred

# Criar ensemble
resnet = load_resnet_model()
efficientnet = load_efficientnet_model()
densenet = load_densenet_model()

ensemble = EnsembleModel([resnet, efficientnet, densenet])
```

#### C. Data Augmentation Avançado

```python
from torchvision import transforms

# Augmentation mais agressivo
advanced_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Mixup (avançado)
def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
```

#### D. Fine-tuning Progressivo

```python
# Fase 1: Treinar apenas última camada
for param in model.parameters():
    param.requires_grad = False

for param in model.fc.parameters():
    param.requires_grad = True

optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
# Treinar por 10 épocas

# Fase 2: Descongelar últimas camadas da ResNet
for param in model.layer4.parameters():
    param.requires_grad = True

optimizer = optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 0.0001},
    {'params': model.fc.parameters(), 'lr': 0.001}
])
# Treinar por mais 10 épocas

# Fase 3: Fine-tuning completo
for param in model.parameters():
    param.requires_grad = True

optimizer = optim.Adam(model.parameters(), lr=0.00001)
# Treinar até convergir
```

### 2. Otimização de Velocidade

#### A. Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for images, labels in train_loader:
    optimizer.zero_grad()
    
    # Forward pass com mixed precision
    with autocast():
        outputs = model(images)
        loss = criterion(outputs, labels)
    
    # Backward pass com scaler
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

#### B. Gradient Accumulation

```python
accumulation_steps = 4  # Simula batch size 4x maior

for i, (images, labels) in enumerate(train_loader):
    outputs = model(images)
    loss = criterion(outputs, labels) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

#### C. DataLoader Otimizado

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=8,        # Aumentar workers
    pin_memory=True,      # Otimização para GPU
    prefetch_factor=2,    # Pré-carregar batches
    persistent_workers=True  # Manter workers vivos
)
```

### 3. Técnicas de Regularização

#### A. Label Smoothing

```python
class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
    
    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        smooth_label = torch.full_like(pred, self.smoothing / (self.num_classes - 1))
        smooth_label.scatter_(1, target.unsqueeze(1), confidence)
        
        log_probs = torch.log_softmax(pred, dim=1)
        loss = -torch.sum(smooth_label * log_probs, dim=1)
        return loss.mean()

criterion = LabelSmoothingLoss(num_classes=7, smoothing=0.1)
```

#### B. Dropout Adaptativo

```python
# Ajustar dropout durante treinamento
class AdaptiveDropout(nn.Module):
    def __init__(self, initial_p=0.5):
        super().__init__()
        self.p = initial_p
    
    def forward(self, x):
        return F.dropout(x, p=self.p, training=self.training)
    
    def reduce_dropout(self, factor=0.9):
        self.p *= factor
```

#### C. Weight Decay com AdamW

```python
# AdamW implementa weight decay corretamente
optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01,  # L2 regularization
    betas=(0.9, 0.999)
)
```

## 📈 Monitoramento e Debugging

### 1. Tensorboard Integration

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/dermatonet_experiment')

for epoch in range(epochs):
    # ... treinamento ...
    
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)
    
    # Adicionar imagens
    writer.add_images('Predictions', images[:8], epoch)
    
    # Adicionar histogramas de pesos
    for name, param in model.named_parameters():
        writer.add_histogram(name, param, epoch)

writer.close()

# Visualizar: tensorboard --logdir=runs
```

### 2. Early Stopping

```python
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

early_stopping = EarlyStopping(patience=10)

for epoch in range(epochs):
    # ... treinamento ...
    early_stopping(val_loss)
    
    if early_stopping.early_stop:
        print(f"Early stopping at epoch {epoch}")
        break
```

### 3. Learning Rate Finder

```python
def find_lr(model, train_loader, criterion, optimizer, device, 
            start_lr=1e-7, end_lr=10, num_iter=100):
    """Encontra a melhor learning rate"""
    
    lr_mult = (end_lr / start_lr) ** (1/num_iter)
    lr = start_lr
    optimizer.param_groups[0]['lr'] = lr
    
    losses = []
    lrs = []
    
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        if i >= num_iter:
            break
        
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        lrs.append(lr)
        
        lr *= lr_mult
        optimizer.param_groups[0]['lr'] = lr
    
    # Plotar
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate Finder')
    plt.show()
    
    return lrs, losses
```

## 🔧 Deploy e Produção

### 1. Otimização do Modelo para Produção

```python
# TorchScript (mais rápido)
model.eval()
example = torch.rand(1, 3, 224, 224).to(device)
traced_model = torch.jit.trace(model, example)
traced_model.save('dermatonet_traced.pt')

# Carregar modelo traced
model = torch.jit.load('dermatonet_traced.pt')
```

### 2. Quantização (reduzir tamanho)

```python
# Quantização dinâmica
import torch.quantization as quantization

quantized_model = quantization.quantize_dynamic(
    model, 
    {nn.Linear}, 
    dtype=torch.qint8
)

# Modelo ficará ~4x menor
torch.save(quantized_model.state_dict(), 'model_quantized.pth')
```

### 3. ONNX Export (interoperabilidade)

```python
import torch.onnx

dummy_input = torch.randn(1, 3, 224, 224).to(device)
torch.onnx.export(
    model,
    dummy_input,
    "dermatonet.onnx",
    export_params=True,
    opset_version=11,
    input_names=['input'],
    output_names=['output']
)
```

### 4. Docker para Deploy

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
```

## 📊 Métricas Avançadas

### 1. ROC-AUC para cada classe

```python
from sklearn.metrics import roc_auc_score, roc_curve

# One-vs-Rest
y_true_binary = label_binarize(all_labels, classes=range(7))
y_scores = model_probabilities  # Probabilidades do modelo

# AUC para cada classe
for i, class_name in enumerate(class_names):
    auc = roc_auc_score(y_true_binary[:, i], y_scores[:, i])
    print(f"{class_name}: AUC = {auc:.4f}")
```

### 2. Grad-CAM (explicabilidade)

```python
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def __call__(self, x, class_idx):
        self.model.zero_grad()
        output = self.model(x)
        
        # Backward pass
        output[0, class_idx].backward()
        
        # Grad-CAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        return cam
```

## 🎯 Checklist de Boas Práticas

- [ ] **Dados**
  - [ ] Dataset balanceado ou com pesos de classe
  - [ ] Data augmentation apropriado
  - [ ] Validação cruzada k-fold
  - [ ] Análise de outliers

- [ ] **Modelo**
  - [ ] Transfer learning aplicado
  - [ ] Arquitetura apropriada para o problema
  - [ ] Regularização adequada (dropout, weight decay)
  - [ ] Ensemble de modelos considerado

- [ ] **Treinamento**
  - [ ] Learning rate apropriada (lr finder)
  - [ ] Early stopping implementado
  - [ ] Checkpoint dos melhores modelos
  - [ ] Mixed precision training (GPU)

- [ ] **Avaliação**
  - [ ] Múltiplas métricas (accuracy, F1, AUC)
  - [ ] Matriz de confusão analisada
  - [ ] Erros qualitativos inspecionados
  - [ ] Teste em dados não vistos

- [ ] **Produção**
  - [ ] Modelo otimizado (TorchScript/ONNX)
  - [ ] API REST documentada
  - [ ] Monitoramento de performance
  - [ ] Logging de predições

## 📚 Recursos Adicionais

### Artigos Científicos
- Esteva et al. (2017) - "Dermatologist-level classification of skin cancer"
- Tschandl et al. (2018) - "The HAM10000 dataset"

### Cursos e Tutoriais
- Fast.ai - Practical Deep Learning for Coders
- PyTorch Official Tutorials
- Stanford CS231n - CNNs for Visual Recognition

### Ferramentas
- Weights & Biases - Experiment tracking
- Gradio - Interface rápida para demos
- SHAP - Explicabilidade de modelos

---

**Boa sorte com seu projeto! 🚀**