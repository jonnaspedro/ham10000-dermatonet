import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import joblib
import argparse
import sys

# Configurações
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 224

# Descrições das classes
CLASS_INFO = {
    'akiec': {
        'name': 'Ceratose Actínica (Actinic Keratoses)',
        'risk': 'MÉDIO',
        'description': 'Lesões pré-cancerosas causadas por exposição solar excessiva'
    },
    'bcc': {
        'name': 'Carcinoma Basocelular (Basal Cell Carcinoma)',
        'risk': 'ALTO',
        'description': 'Tipo mais comum de câncer de pele'
    },
    'bkl': {
        'name': 'Ceratose Benigna (Benign Keratosis)',
        'risk': 'BAIXO',
        'description': 'Lesões benignas comuns, não cancerosas'
    },
    'df': {
        'name': 'Dermatofibroma',
        'risk': 'BAIXO',
        'description': 'Nódulo benigno de tecido fibroso na pele'
    },
    'mel': {
        'name': 'Melanoma',
        'risk': 'MUITO ALTO',
        'description': 'Tipo mais perigoso de câncer de pele'
    },
    'nv': {
        'name': 'Nevo Melanocítico (Melanocytic Nevi)',
        'risk': 'BAIXO',
        'description': 'Pintas comuns, geralmente benignas'
    },
    'vasc': {
        'name': 'Lesão Vascular (Vascular Lesions)',
        'risk': 'BAIXO',
        'description': 'Lesões relacionadas a vasos sanguíneos'
    }
}

def load_model(model_path='dermatonet_best.pth', metadata_path='model_metadata.pkl'):
    """Carrega o modelo treinado"""
    print("Carregando modelo...")
    
    # Carregar metadados
    try:
        metadata = joblib.load(metadata_path)
        class_names = metadata['class_names']
        print("✅ Metadados carregados")
    except FileNotFoundError:
        print(f"❌ Erro: Arquivo {metadata_path} não encontrado!")
        sys.exit(1)
    
    # Criar arquitetura
    model = models.resnet50(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, len(class_names))
    )
    
    # Carregar pesos
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model = model.to(DEVICE)
        model.eval()
        print("✅ Modelo carregado com sucesso")
        print(f"   Dispositivo: {DEVICE}")
    except FileNotFoundError:
        print(f"❌ Erro: Arquivo {model_path} não encontrado!")
        print("   Execute train_model.py primeiro para treinar o modelo.")
        sys.exit(1)
    
    return model, class_names, metadata

def preprocess_image(image_path):
    """Pré-processa imagem para inferência"""
    try:
        image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"❌ Erro: Imagem '{image_path}' não encontrada!")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Erro ao carregar imagem: {e}")
        sys.exit(1)
    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return transform(image).unsqueeze(0).to(DEVICE), image

def predict(model, image_tensor, class_names, top_k=3):
    """Realiza predição"""
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Top-K predições
        top_probs, top_indices = torch.topk(probabilities, top_k)
        
        predictions = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            class_name = class_names[idx.item()]
            predictions.append({
                'class': class_name,
                'confidence': prob.item() * 100,
                'info': CLASS_INFO[class_name]
            })
    
    return predictions

def print_results(predictions, verbose=False):
    """Imprime resultados formatados"""
    print("\n" + "="*70)
    print("🔬 RESULTADO DA ANÁLISE")
    print("="*70)
    
    # Predição principal
    pred = predictions[0]
    info = pred['info']
    
    print("\nDIAGNÓSTICO PREDITO")
    print(f"   Classe: {info['name']}")
    print(f"   Confiança: {pred['confidence']:.2f}%")
    print(f"   Nível de Risco: {info['risk']}")
    print(f"   Descrição: {info['description']}")
    
    # Recomendação
    print("\nRECOMENDAÇÃO")
    if info['risk'] in ['MUITO ALTO', 'ALTO']:
        print("   URGENTE: Procure um dermatologista IMEDIATAMENTE!")
    elif info['risk'] == 'MÉDIO':
        print("   Consulte um dermatologista para avaliação.")
    else:
        print("   Acompanhamento de rotina recomendado.")
    
    # Outras possibilidades (verbose)
    if verbose and len(predictions) > 1:
        print(f"\n📊 OUTRAS POSSIBILIDADES (Top-{len(predictions)})")
        for i, pred in enumerate(predictions[1:], 2):
            info = pred['info']
            print(f"   {i}. {info['name']}: {pred['confidence']:.2f}%")
    
    # Aviso médico
    print("\nAVISO IMPORTANTE")
    print("   Este resultado é gerado por IA e NÃO substitui diagnóstico médico.")
    print("   Sempre consulte um profissional de saúde qualificado.")
    
    print("="*70 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description='DermatoNet - Classificação de Lesões de Pele',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python inference.py imagem.jpg
  python inference.py imagem.jpg --verbose
  python inference.py imagem.jpg --top-k 5
  python inference.py imagem.jpg --model meu_modelo.pth
        """
    )
    
    parser.add_argument('image', type=str, help='Caminho para a imagem')
    parser.add_argument('--model', type=str, default='dermatonet_best.pth',
                        help='Caminho para o modelo (.pth)')
    parser.add_argument('--metadata', type=str, default='model_metadata.pkl',
                        help='Caminho para os metadados (.pkl)')
    parser.add_argument('--top-k', type=int, default=3,
                        help='Número de predições a mostrar (padrão: 3)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Mostrar informações detalhadas')
    
    args = parser.parse_args()
    
    # Banner
    print("\n" + "="*70)
    print("🔬 DERMATONET - SISTEMA DE CLASSIFICAÇÃO DE LESÕES DE PELE")
    print("="*70)
    
    # Carregar modelo
    model, class_names, metadata = load_model(args.model, args.metadata)
    
    # Informações do modelo (verbose)
    if args.verbose:
        print("\nInformações do Modelo:")
        print(f"   Acurácia de Validação: {metadata['best_val_acc']:.2f}%")
        print(f"   Acurácia de Teste: {metadata['test_acc']:.2f}%")
        print(f"   Classes: {', '.join(class_names)}")
    
    # Pré-processar imagem
    print(f"\nProcessando imagem: {args.image}")
    image_tensor, original_image = preprocess_image(args.image)
    print(f"   Tamanho original: {original_image.size}")
    print(f"   Redimensionada para: {IMG_SIZE}x{IMG_SIZE}")
    
    # Predição
    print("\nAnalisando...")
    predictions = predict(model, image_tensor, class_names, top_k=args.top_k)
    
    # Mostrar resultados
    print_results(predictions, verbose=args.verbose)

if __name__ == "__main__":
    main()