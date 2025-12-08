# 🔧 DermatoNet - Guia de Solução de Problemas

## 🚨 Problemas Comuns e Soluções

### 1. Erro ao Baixar o Dataset

#### Problema: "kaggle.ApiException: Unauthorized"
```
kaggle.rest.ApiException: (401)
Reason: Unauthorized
```

**Solução:**
```bash
# 1. Certifique-se de ter uma conta no Kaggle
# 2. Vá em kaggle.com → Account → API → Create New API Token
# 3. Salve kaggle.json no local correto:

# Windows
mkdir %USERPROFILE%\.kaggle
move kaggle.json %USERPROFILE%\.kaggle\

# Linux/Mac
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

#### Problema: Download lento ou timeout
**Solução:**
```python
# Aumentar timeout no código
import kagglehub
kagglehub.config.timeout = 600  # 10 minutos
```

---

### 2. Problemas de Memória (CUDA Out of Memory)

#### Erro: "RuntimeError: CUDA out of memory"

**Solução 1: Reduzir Batch Size**
```python
# No train_model.py, linha ~23
BATCH_SIZE = 16  # Era 32, reduzir para 16 ou 8
```

**Solução 2: Gradient Checkpointing**
```python
from torch.utils.checkpoint import checkpoint

# No forward pass do modelo
def forward(self, x):
    x = checkpoint(self.conv1, x)
    x = checkpoint(self.conv2, x)
    return x
```

**Solução 3: Limpar Cache**
```python
# Adicionar após cada época
import torch
torch.cuda.empty_cache()
```

**Solução 4: Mixed Precision**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(images)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

### 3. Modelo Não Converge (Loss Alta)

#### Problema: Loss não diminui ou oscila muito

**Diagnóstico:**
```python
# Verificar learning rate
print(f"Learning rate atual: {optimizer.param_groups[0]['lr']}")

# Verificar gradientes
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm = {param.grad.norm()}")
```

**Solução 1: Ajustar Learning Rate**
```python
# Learning rate muito alta
LEARNING_RATE = 0.00001  # Era 0.001

# Ou usar scheduler mais agressivo
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    patience=2,  # Era 3
    factor=0.3   # Era 0.5
)
```

**Solução 2: Verificar Normalização**
```python
# Garantir que está usando a normalização correta
transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet
```

**Solução 3: Verificar Desbalanceamento**
```python
# Aumentar peso das classes minoritárias
class_weights = 1.0 / torch.tensor([...], dtype=torch.float)
class_weights = class_weights / class_weights.sum() * len(CLASSES)
class_weights = class_weights * 2  # Aumentar penalidade
```

---

### 4. Overfitting (Alta Acurácia no Treino, Baixa na Validação)

#### Sintomas:
- Train Acc > 95%, Val Acc < 75%
- Loss de treino muito baixa, loss de validação alta

**Solução 1: Aumentar Dropout**
```python
model.fc = nn.Sequential(
    nn.Dropout(0.7),  # Era 0.5
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),  # Era 0.3
    nn.Linear(512, num_classes)
)
```

**Solução 2: Mais Data Augmentation**
```python
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(30),  # Era 20
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # NOVO
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  # NOVO
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

**Solução 3: Early Stopping**
```python
# Parar quando validação não melhora por N épocas
best_val_loss = float('inf')
patience = 5
counter = 0

for epoch in range(EPOCHS):
    # ... treinamento ...
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
```

---

### 5. Underfitting (Acurácia Baixa no Treino e Validação)

#### Sintomas:
- Train Acc < 60%, Val Acc < 60%
- Loss não diminui significativamente

**Solução 1: Descongelar Mais Camadas**
```python
# Descongelar toda a ResNet, não apenas a última camada
for param in model.parameters():
    param.requires_grad = True

# Usar learning rates diferentes
optimizer = optim.Adam([
    {'params': model.layer3.parameters(), 'lr': 0.0001},
    {'params': model.layer4.parameters(), 'lr': 0.0001},
    {'params': model.fc.parameters(), 'lr': 0.001}
])
```

**Solução 2: Aumentar Capacidade do Modelo**
```python
# Usar modelo maior
model = models.resnet101(pretrained=True)  # Era resnet50

# Ou adicionar mais camadas fully connected
model.fc = nn.Sequential(
    nn.Linear(num_features, 1024),  # Era 512
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, num_classes)
)
```

**Solução 3: Treinar por Mais Épocas**
```python
EPOCHS = 50  # Era 30
```

---

### 6. Problemas com Streamlit

#### Erro: "Module not found"
```bash
# Reinstalar dependências
pip install -r requirements.txt --force-reinstall
```

#### Erro: "Model file not found"
```python
# Verificar se os arquivos existem
import os
print(os.path.exists('dermatonet_best.pth'))
print(os.path.exists('model_metadata.pkl'))

# Se não existem, treinar o modelo primeiro
python train_model.py
```

#### Aplicação não carrega ou fica lenta
```python
# No app.py, otimizar cache
@st.cache_resource(show_spinner=False)
def load_model():
    # ... código existente ...
    pass

# Reduzir tamanho das imagens antes de processar
def resize_image(image, max_size=800):
    image.thumbnail((max_size, max_size))
    return image
```

---

### 7. Erros de Compatibilidade

#### Problema: Versões incompatíveis do PyTorch

**Solução:**
```bash
# Desinstalar versões antigas
pip uninstall torch torchvision torchaudio

# Instalar versão compatível
# Para CPU
pip install torch torchvision torchaudio

# Para GPU (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Para GPU (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Problema: Conflito com NumPy/Pandas
```bash
pip install numpy==1.24.0 pandas==2.0.0 --force-reinstall
```

---

### 8. Dataset Corrompido ou Incompleto

#### Verificar integridade das imagens
```python
from PIL import Image
import os

def verify_images(img_dir):
    corrupted = []
    for filename in os.listdir(img_dir):
        if filename.endswith('.jpg'):
            try:
                img = Image.open(os.path.join(img_dir, filename))
                img.verify()
            except Exception as e:
                corrupted.append(filename)
                print(f"Corrupted: {filename}")
    
    return corrupted

corrupted = verify_images(img_dir1)
print(f"Total corrupted: {len(corrupted)}")
```

#### Baixar dataset manualmente
```bash
# Se kagglehub falhar, baixar manualmente:
# 1. Ir em https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
# 2. Clicar em "Download"
# 3. Extrair na pasta do projeto
```

---

### 9. Problemas de Performance

#### Treinamento muito lento

**Solução 1: Verificar GPU**
```python
import torch

print(f"CUDA disponível: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Verificar se modelo está na GPU
print(f"Modelo no dispositivo: {next(model.parameters()).device}")
```

**Solução 2: Aumentar num_workers**
```python
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,  # Era 2
    pin_memory=True,
    prefetch_factor=2
)
```

**Solução 3: Usar GPU mais potente no Colab**
```python
# No Colab: Runtime > Change runtime type > GPU > A100
```

---

### 10. Resultados Inconsistentes

#### Garantir reprodutibilidade
```python
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Chamar no início do script
set_seed(42)
```

---

## 📊 Checklist de Debug

Quando encontrar problemas, verifique:

- [ ] **Ambiente**
  - [ ] Python 3.8+
  - [ ] PyTorch instalado corretamente
  - [ ] CUDA funcional (se usando GPU)
  - [ ] Espaço em disco suficiente (>5GB)

- [ ] **Dataset**
  - [ ] Download completo
  - [ ] Kaggle credentials configuradas
  - [ ] Imagens não corrompidas
  - [ ] Metadados corretos

- [ ] **Modelo**
  - [ ] Arquivo .pth existe
  - [ ] Arquivo .pkl existe
  - [ ] Arquitetura compatível
  - [ ] Dimensões corretas

- [ ] **Treinamento**
  - [ ] Batch size apropriado
  - [ ] Learning rate adequada
  - [ ] Memória GPU suficiente
  - [ ] Augmentation não muito agressivo

- [ ] **Código**
  - [ ] Sem erros de sintaxe
  - [ ] Imports corretos
  - [ ] Paths corretos
  - [ ] Indentação correta

---

## 🆘 Ainda com Problemas?

### 1. Ativar Modo Verbose
```python
# Adicionar no início do script
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 2. Criar Issue no GitHub
Inclua:
- Descrição detalhada do problema
- Mensagem de erro completa
- Sistema operacional
- Versões do Python e PyTorch
- Steps para reproduzir

### 3. Consultar Documentação
- PyTorch: https://pytorch.org/docs/
- Streamlit: https://docs.streamlit.io/
- Kaggle API: https://github.com/Kaggle/kaggle-api

### 4. Comunidade
- PyTorch Forum: https://discuss.pytorch.org/
- Stack Overflow: Tag [pytorch]
- Reddit: r/MachineLearning

---

## 💡 Dicas Gerais

1. **Sempre verifique os logs** - A maioria dos erros está clara nos logs
2. **Teste incrementalmente** - Não mude muitas coisas de uma vez
3. **Mantenha backups** - Salve modelos intermediários
4. **Use controle de versão** - Git para rastrear mudanças
5. **Documente soluções** - Mantenha um log de problemas resolvidos

---

**Boa sorte! 🍀**