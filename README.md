# DermatoNet-HAM10000

Projeto DermatoNet — um sistema de classificação automática de lesões de pele utilizando Deep Learning e o dataset HAM10000. O objetivo é replicar experimentos de artigos científicos, treinar modelos em Google Colab e desenvolver uma aplicação em Streamlit capaz de classificar imagens dermatológicas e registrar interações em banco de dados.

## Integrantes do Grupo
- Jonnas Pedro  
- Laysa Marina  
- Cauã Rocha  
- Vitor Farias  

## Conteúdo do Projeto
- Scripts de pré-processamento e treinamento  
- Experimentos baseados em artigos científicos  
- Aplicação Web com Streamlit  
- Banco de dados SQLite/MySQL para registro das interações  
- Artigo científico em desenvolvimento  

## Dataset
- [**Skin Cancer MNIST: HAM10000 - a large collection of multi-source dermatoscopic images of pigmented lesions**](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

## Objetivo
Desenvolver um modelo de deep learning capaz de classificar diferentes tipos de lesões de pele e disponibilizar essa solução em uma aplicação na nuvem.

## Status
- Experimentação inicial em andamento  
- Artigo em escrita  
- Aplicação em desenvolvimento

## Como rodar
- **OBS:** Mais jeitos de rodar os códigos em [run](run.py), e antes de instalar o dataset, leia [dataset_install](src/dataset_install.py)
```bash
# Instale as dependências
pip install -r requirements.txt

# Para instalar o dataset
python run.py install

# Para treinar
python run.py train

# Para iniciar o streamlit (CLI)
streamlit streamlit_app.py
```