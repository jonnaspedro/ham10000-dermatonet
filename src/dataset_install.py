"""
ham10000-dermatonet/
    dataset/
    generated/
    src/
"""

import kagglehub

path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")

print("Path to dataset files:", path)