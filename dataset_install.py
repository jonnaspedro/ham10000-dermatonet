# Para o programa funcionar corretamente, certifique-se de colocar os arquivos
# baixados pelo Kagglehub na pasta './dataset'

import kagglehub

# Download latest version
path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")

print("Path to dataset files:", path)