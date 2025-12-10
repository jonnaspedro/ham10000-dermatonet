import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import sqlite3
import joblib

st.set_page_config(
    page_title="DermatoNet - Classificação de Lesões de Pele",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        padding: 1rem 0;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        padding-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .warning-box {
        background-color: #d8a80f;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 224

CLASS_DESCRIPTIONS = {
    'akiec': {
        'name': 'Ceratose Actínica',
        'description': 'Lesões pré-cancerosas causadas por exposição solar excessiva.',
        'risk': 'Médio - Pode evoluir para câncer de pele',
        'recommendation': 'Consulte um dermatologista para avaliação e possível tratamento.'
    },
    'bcc': {
        'name': 'Carcinoma Basocelular',
        'description': 'Tipo mais comum de câncer de pele, geralmente causado por exposição solar.',
        'risk': 'Alto - É um tipo de câncer de pele',
        'recommendation': 'URGENTE: Procure um dermatologista imediatamente.'
    },
    'bkl': {
        'name': 'Ceratose Benigna',
        'description': 'Lesões benignas comuns, não cancerosas.',
        'risk': 'Baixo - Geralmente inofensivo',
        'recommendation': 'Acompanhamento de rotina com dermatologista.'
    },
    'df': {
        'name': 'Dermatofibroma',
        'description': 'Nódulo benigno de tecido fibroso na pele.',
        'risk': 'Baixo - Lesão benigna',
        'recommendation': 'Acompanhamento opcional. Consulte dermatologista se houver mudanças.'
    },
    'mel': {
        'name': 'Melanoma',
        'description': 'Tipo mais perigoso de câncer de pele.',
        'risk': 'MUITO ALTO - Câncer agressivo',
        'recommendation': 'URGENTE: Procure um oncologista/dermatologista IMEDIATAMENTE!'
    },
    'nv': {
        'name': 'Nevo Melanocítico',
        'description': 'Pintas comuns, geralmente benignas.',
        'risk': 'Baixo - Geralmente benigno',
        'recommendation': 'Acompanhamento de rotina. Atenção a mudanças de tamanho/cor.'
    },
    'vasc': {
        'name': 'Lesão Vascular',
        'description': 'Lesões relacionadas a vasos sanguíneos.',
        'risk': 'Baixo - Geralmente benigno',
        'recommendation': 'Consulte dermatologista para avaliação.'
    }
}

def init_database():
    """Cria banco de dados SQLite para registro de interações"""
    conn = sqlite3.connect('generated/dermatonet_logs.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            predicted_class TEXT,
            confidence REAL,
            image_name TEXT,
            user_feedback TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

def log_prediction(predicted_class, confidence, image_name, feedback=None):
    """Registra predição no banco de dados"""
    conn = sqlite3.connect('dermatonet_logs.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO predictions (predicted_class, confidence, image_name, user_feedback)
        VALUES (?, ?, ?, ?)
    ''', (predicted_class, confidence, image_name, feedback))
    
    conn.commit()
    conn.close()

def get_statistics():
    """Obtém estatísticas do banco de dados"""
    conn = sqlite3.connect('dermatonet_logs.db')
    df = pd.read_sql_query("SELECT * FROM predictions", conn)
    conn.close()
    return df

@st.cache_resource
def load_model():
    """Carrega modelo treinado"""
    try:
        metadata = joblib.load('generated/model_metadata.pkl')
        class_names = metadata['class_names']
        
        model = models.resnet50(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, len(class_names))
        )
        
        model.load_state_dict(torch.load('generated/dermatonet_best.pth', map_location=DEVICE))
        model = model.to(DEVICE)
        model.eval()
        
        return model, class_names, metadata
    except Exception as e:
        st.error(f"❌ Erro ao carregar modelo: {e}")
        return None, None, None

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(image, model, class_names):
    """Realiza predição em uma imagem"""
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_class = class_names[predicted.item()]
    confidence_value = confidence.item() * 100
    
    all_probs = probabilities[0].cpu().numpy()
    prob_dict = {class_names[i]: float(all_probs[i] * 100) for i in range(len(class_names))}
    
    return predicted_class, confidence_value, prob_dict

def main():
    init_database()
    
    st.markdown('<div class="main-header">🔬 DermatoNet</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Sistema Inteligente de Classificação de Lesões de Pele</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ AVISO MÉDICO IMPORTANTE:</strong><br>
        Este sistema é uma ferramenta de apoio e NÃO substitui o diagnóstico médico profissional.
        Sempre consulte um dermatologista qualificado para diagnóstico e tratamento adequados.
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("📋 Menu")
        page = st.radio("Navegação", ["🏠 Classificação", "📊 Estatísticas", "ℹ️ Sobre"])
        
        st.markdown("---")
        st.markdown("### 🎯 Precisão do Modelo")
        
        model, class_names, metadata = load_model()
        
        if metadata:
            st.metric("Acurácia de Teste", f"{metadata['test_acc']:.2f}%")
            st.metric("Melhor Val. Acc", f"{metadata['best_val_acc']:.2f}%")
        
        st.markdown("---")
        st.markdown("**Dataset:** HAM10000")
        st.markdown("**Modelo:** ResNet50")
        st.markdown("**Transfer Learning:** ✅")
    
    if page == "🏠 Classificação":
        if model is None:
            st.error("❌ Modelo não encontrado. Execute o script de treinamento primeiro.")
            return
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📤 Upload da Imagem")
            uploaded_file = st.file_uploader(
                "Escolha uma imagem dermatoscópica",
                type=['jpg', 'jpeg', 'png'],
                help="Formatos aceitos: JPG, JPEG, PNG"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption='Imagem Carregada', use_container_width=True)
                
                if st.button("🔍 Analisar Imagem", type="primary", use_container_width=True):
                    with st.spinner("Analisando imagem..."):
                        predicted_class, confidence, prob_dict = predict_image(image, model, class_names)
                        
                        st.session_state['prediction'] = {
                            'class': predicted_class,
                            'confidence': confidence,
                            'probs': prob_dict,
                            'image_name': uploaded_file.name
                        }
                        
                        log_prediction(predicted_class, confidence, uploaded_file.name)
        
        with col2:
            st.subheader("🎯 Resultado da Análise")
            
            if 'prediction' in st.session_state:
                pred = st.session_state['prediction']
                class_info = CLASS_DESCRIPTIONS[pred['class']]
                
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>🔬 Diagnóstico Predito</h2>
                    <h1>{class_info['name']}</h1>
                    <h3>Confiança: {pred['confidence']:.2f}%</h3>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### 📝 Informações da Lesão")
                st.write(f"**Descrição:** {class_info['description']}")
                
                _risk_colors = {
                    'Baixo': '🟢',
                    'Médio': '🟡',
                    'Alto': '🟠',
                    'MUITO ALTO': '🔴'
                }
                risk_emoji = '🔴' if 'ALTO' in class_info['risk'] else '🟡' if 'Médio' in class_info['risk'] else '🟢'
                st.write(f"**Nível de Risco:** {risk_emoji} {class_info['risk']}")
                
                st.markdown(f"""
                <div class="info-box">
                    <strong>💡 Recomendação:</strong><br>
                    {class_info['recommendation']}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### 📊 Distribuição de Probabilidades")
                
                prob_df = pd.DataFrame({
                    'Classe': [CLASS_DESCRIPTIONS[k]['name'] for k in pred['probs'].keys()],
                    'Probabilidade (%)': list(pred['probs'].values())
                }).sort_values('Probabilidade (%)', ascending=False)
                
                st.bar_chart(prob_df.set_index('Classe'))
            else:
                st.info("👆 Faça upload de uma imagem e clique em 'Analisar' para ver os resultados.")
    
    elif page == "📊 Estatísticas":
        st.subheader("📊 Estatísticas de Uso do Sistema")
        
        df_stats = get_statistics()
        
        if len(df_stats) > 0:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total de Análises", len(df_stats))
            
            with col2:
                st.metric("Confiança Média", f"{df_stats['confidence'].mean():.2f}%")
            
            with col3:
                st.metric("Feedbacks Recebidos", df_stats['user_feedback'].notna().sum())
            
            st.markdown("### 📈 Distribuição de Classes Preditas")
            class_dist = df_stats['predicted_class'].value_counts()
            st.bar_chart(class_dist)
            
            st.markdown("### 📋 Histórico Recente")
            st.dataframe(
                df_stats[['timestamp', 'predicted_class', 'confidence', 'image_name']].tail(10),
                use_container_width=True
            )
        else:
            st.info("📭 Nenhuma análise realizada ainda.")
    
    elif page == "ℹ️ Sobre":
        st.subheader("ℹ️ Sobre o DermatoNet")
        
        st.markdown("""
        ### 🎯 Objetivo
        O **DermatoNet** é um sistema de classificação automática de lesões de pele desenvolvido
        com Deep Learning utilizando o dataset HAM10000.
        
        ### 🧠 Tecnologia
        - **Arquitetura:** ResNet50 com Transfer Learning
        - **Dataset:** HAM10000 (10.015 imagens dermatoscópicas)
        - **Classes:** 7 tipos diferentes de lesões de pele
        - **Framework:** PyTorch
        - **Interface:** Streamlit
        
        ### 📚 Classes Identificadas
        """)
        
        for class_key, info in CLASS_DESCRIPTIONS.items():
            with st.expander(f"🔬 {info['name']}"):
                st.write(f"**Descrição:** {info['description']}")
                st.write(f"**Risco:** {info['risk']}")
                st.write(f"**Recomendação:** {info['recommendation']}")
        
        st.markdown("""
        ### ⚠️ Disclaimer
        Este sistema foi desenvolvido para fins educacionais e de pesquisa. 
        **NÃO** deve ser usado como única fonte para diagnóstico médico.
        Sempre consulte um profissional de saúde qualificado.
        
        ### 👨‍💻 Desenvolvimento
        Projeto desenvolvido utilizando as melhores práticas de Deep Learning
        para classificação de imagens médicas.
        
        ### 📊 Dataset
        **HAM10000:** Human Against Machine with 10000 training images
        - Fonte: Kaggle
        - Link: [HAM10000 Dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
        """)

if __name__ == "__main__":
    main()