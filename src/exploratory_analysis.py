import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*70)
print("ANÁLISE EXPLORATÓRIA DO DATASET HAM10000")
print("="*70)

path = "dataset"

metadata_path = os.path.join(path, "HAM10000_metadata.csv")
df = pd.read_csv(metadata_path)

print("\nINFORMAÇÕES GERAIS")
print(f"{'='*70}")
print(f"Total de registros: {len(df)}")
print(f"Total de colunas: {len(df.columns)}")
print("\nColunas disponíveis:")
for col in df.columns:
    print(f"  - {col}")

print("\nVALORES AUSENTES")
print(f"{'='*70}")
missing = df.isnull().sum()
print(missing[missing > 0])
if missing.sum() == 0:
    print("✅ Nenhum valor ausente encontrado!")

print("\nDISTRIBUIÇÃO DAS CLASSES")
print(f"{'='*70}")

class_names = {
    'akiec': 'Actinic Keratoses',
    'bcc': 'Basal Cell Carcinoma',
    'bkl': 'Benign Keratosis',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic Nevi',
    'vasc': 'Vascular Lesions'
}

class_counts = df['dx'].value_counts()
for cls, count in class_counts.items():
    percentage = (count / len(df)) * 100
    print(f"{class_names[cls]:30s} ({cls}): {count:5d} ({percentage:5.2f}%)")

print("\nGERANDO VISUALIZAÇÕES...")

fig = plt.figure(figsize=(18, 12))

ax1 = plt.subplot(2, 3, 1)
class_counts.plot(kind='bar', ax=ax1, color='skyblue', edgecolor='black')
ax1.set_title('Distribuição das Classes', fontsize=14, fontweight='bold')
ax1.set_xlabel('Classe', fontsize=12)
ax1.set_ylabel('Quantidade', fontsize=12)
ax1.tick_params(axis='x', rotation=45)
for i, v in enumerate(class_counts):
    ax1.text(i, v + 100, str(v), ha='center', va='bottom', fontsize=10)

ax2 = plt.subplot(2, 3, 2)
gender_counts = df['sex'].value_counts()
colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']
ax2.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', 
        startangle=90, colors=colors, explode=(0.05, 0.05, 0))
ax2.set_title('Distribuição por Gênero', fontsize=14, fontweight='bold')

ax3 = plt.subplot(2, 3, 3)
df['age'].hist(bins=30, ax=ax3, color='coral', edgecolor='black', alpha=0.7)
ax3.set_title('Distribuição de Idade', fontsize=14, fontweight='bold')
ax3.set_xlabel('Idade', fontsize=12)
ax3.set_ylabel('Frequência', fontsize=12)
ax3.axvline(df['age'].mean(), color='red', linestyle='--', linewidth=2, label=f'Média: {df["age"].mean():.1f}')
ax3.legend()

ax4 = plt.subplot(2, 3, 4)
localization_counts = df['localization'].value_counts().head(10)
localization_counts.plot(kind='barh', ax=ax4, color='lightgreen', edgecolor='black')
ax4.set_title('Top 10 Localizações', fontsize=14, fontweight='bold')
ax4.set_xlabel('Quantidade', fontsize=12)
ax4.set_ylabel('Localização', fontsize=12)

ax5 = plt.subplot(2, 3, 5)
gender_class = pd.crosstab(df['dx'], df['sex'])
gender_class.plot(kind='bar', ax=ax5, stacked=False)
ax5.set_title('Classes por Gênero', fontsize=14, fontweight='bold')
ax5.set_xlabel('Classe', fontsize=12)
ax5.set_ylabel('Quantidade', fontsize=12)
ax5.tick_params(axis='x', rotation=45)
ax5.legend(title='Gênero')

ax6 = plt.subplot(2, 3, 6)
class_loc = pd.crosstab(df['dx'], df['localization'])
sns.heatmap(class_loc, annot=False, cmap='YlOrRd', ax=ax6, cbar_kws={'label': 'Contagem'})
ax6.set_title('Heatmap: Classe vs Localização', fontsize=14, fontweight='bold')
ax6.set_xlabel('Localização', fontsize=12)
ax6.set_ylabel('Classe', fontsize=12)
plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('generated/eda_visualizations.png', dpi=300, bbox_inches='tight')
print("✅ Visualizações salvas em: generated/eda_visualizations.png")

print("\nESTATÍSTICAS DESCRITIVAS")
print(f"{'='*70}")
print(df.describe())

print("\nINFORMAÇÕES ADICIONAIS")
print(f"{'='*70}")
print(f"Idade média: {df['age'].mean():.2f} anos")
print(f"Idade mínima: {df['age'].min():.0f} anos")
print(f"Idade máxima: {df['age'].max():.0f} anos")
print(f"Desvio padrão da idade: {df['age'].std():.2f} anos")

print(f"\nLocalizações únicas: {df['localization'].nunique()}")
print(f"Tipos de diagnóstico: {df['dx_type'].nunique()}")

print("\nVERIFICANDO IMAGENS")
print(f"{'='*70}")

img_dir1 = os.path.join(path, "HAM10000_images_part_1")
img_dir2 = os.path.join(path, "HAM10000_images_part_2")

images_part1 = len([f for f in os.listdir(img_dir1) if f.endswith('.jpg')])
images_part2 = len([f for f in os.listdir(img_dir2) if f.endswith('.jpg')])

print(f"Imagens em part_1: {images_part1}")
print(f"Imagens em part_2: {images_part2}")
print(f"Total de imagens: {images_part1 + images_part2}")

print("\nGERANDO EXEMPLOS DE IMAGENS POR CLASSE")
print(f"{'='*70}")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for idx, (cls, cls_name) in enumerate(class_names.items()):
    if idx >= 7:
        axes[idx].axis('off')
        continue
    
    sample = df[df['dx'] == cls].iloc[0]
    img_id = sample['image_id']
    
    img_path1 = os.path.join(img_dir1, f"{img_id}.jpg")
    img_path2 = os.path.join(img_dir2, f"{img_id}.jpg")
    
    img_path = img_path1 if os.path.exists(img_path1) else img_path2
    
    if os.path.exists(img_path):
        img = Image.open(img_path)
        axes[idx].imshow(img)
        axes[idx].set_title(f"{cls_name}\n({cls})", fontsize=10, fontweight='bold')
        axes[idx].axis('off')

axes[7].axis('off')

plt.tight_layout()
plt.savefig('generated/class_examples.png', dpi=300, bbox_inches='tight')
print("✅ Exemplos de imagens salvos em: generated/class_examples.png")

print("\nANÁLISE DE DESBALANCEAMENTO")
print(f"{'='*70}")

max_class = class_counts.max()
min_class = class_counts.min()
imbalance_ratio = max_class / min_class

print(f"Classe mais frequente: {class_counts.idxmax()} ({max_class} imagens)")
print(f"Classe menos frequente: {class_counts.idxmin()} ({min_class} imagens)")
print(f"Razão de desbalanceamento: {imbalance_ratio:.2f}x")

if imbalance_ratio > 10:
    print("⚠️ ATENÇÃO: Dataset altamente desbalanceado!")
    print("   Recomendações:")
    print("   - Usar pesos de classe no loss")
    print("   - Considerar data augmentation")
    print("   - Avaliar métricas além de accuracy (F1-score, recall)")
else:
    print("✅ Desbalanceamento moderado - pesos de classe devem ajudar")

print(f"\n{'='*70}")
print("📋 RESUMO DA ANÁLISE EXPLORATÓRIA")
print(f"{'='*70}")
print("✅ Dataset carregado com sucesso")
print(f"✅ {len(df)} imagens analisadas")
print(f"✅ {len(class_names)} classes diferentes")
print("✅ Visualizações geradas:")
print("   - eda_visualizations.png")
print("   - class_examples.png")
print("\nPróximo passo: Execute train_model.py para treinar o modelo!")
print(f"{'='*70}")

plt.show()