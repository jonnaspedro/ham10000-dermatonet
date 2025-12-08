import subprocess
import sys

if len(sys.argv) < 2:
    print("Use: python run.py [install|analysis|train|interface|startall|interfaceall]")
    sys.exit(1)

option = sys.argv[1]

scripts = {
    "install": ["src/dataset_install.py"],
    "analysis": ["src/exploratory_analysis.py"],
    "train": ["src/train_model.py"],
    "interface": ["src/exploratory_analysis.py", "src/train_model.py", "src/inference.py"],
    "startall": ["src/exploratory_analysis.py", "src/train_model.py"],
    "interfaceall": ["src/exploratory_analysis.py", "src/train_model.py", "src/inference.py"],
}

for s in scripts[option]:
    subprocess.run(["python", s], check=True)
