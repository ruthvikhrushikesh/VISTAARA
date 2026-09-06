"""
VISTAARA - one-time setup script.

Run this BEFORE app.py, once per machine: 
    python install_dependencies.py

This only installs Python packages. It does NOT download the AI model -
the model already ships alongside this script inside the
SEN2SRLite_RGBN_x4 folder, so nothing extra gets pulled from the internet
for that part.

Uses the CPU-only build of PyTorch (no GPU required / no CUDA download).
"""

import subprocess
import sys
import os

# CPU-only torch build - smaller download, works with no GPU
TORCH_CMD = [
    sys.executable, "-m", "pip", "install",
    "torch", "--index-url", "https://download.pytorch.org/whl/cpu"
]

# Everything else the app / model needs
PACKAGES = [
    "streamlit",
    "numpy",
    "rasterio",
    "matplotlib",
    "Pillow",
    "plotly",
    "streamlit-image-comparison",
    "mlstac",
    "safetensors",
    "sen2sr",
    "pandas",
]

MODEL_FOLDER = os.path.join(
    "SEN2SRLite_RGBN_x4", "SEN2SRLite", "NonReference_RGBN_x4"
)


def run(cmd, label):
    print(f"\n--- Installing {label} ---")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[FAILED] Could not install {label}.")
        print("Scroll up to see the actual pip error, fix that, then re-run this script.")
        sys.exit(1)


def check_model_folder():
    if not os.path.isdir(MODEL_FOLDER):
        print(f"\n[WARNING] Model folder not found at: {MODEL_FOLDER}")
        print("Make sure the SEN2SRLite_RGBN_x4 folder sits next to this script")
        print("(it should have come bundled with whatever sent you this project).")
    else:
        print(f"\n[OK] Found model folder at: {MODEL_FOLDER}")


def main():
    print("=" * 60)
    print("VISTAARA setup - installing required Python packages")
    print("(CPU-only PyTorch build - no GPU needed)")
    print("=" * 60)

    run(TORCH_CMD, "PyTorch (CPU build)")
    run([sys.executable, "-m", "pip", "install"] + PACKAGES, "remaining packages")

    check_model_folder()

    print("\n" + "=" * 60)
    print("All set! Now run:")
    print("    streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
