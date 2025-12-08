import subprocess
import sys

if len(sys.argv) < 2:
    print("Use: python runner.py [install|train|streamlit|start]")
    sys.exit(1)

option = sys.argv[1]

scripts = {
    "install": ["dataset_install.py"],
    "train": ["train_model.py"],
    "streamlit": ["streamlit_app.py"],
    "start": ["train_model.py", "streamlit_app.py"],
}

for s in scripts[option]:
    subprocess.run(["python", s], check=True)
