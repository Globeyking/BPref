#!/bin/bash

set -e

ENV_NAME="bpref"

echo "=== Creating conda environment ==="
conda env create -f conda_env.yml || echo "Environment already exists"
conda activate $ENV_NAME

# Upgrade pip/setuptools/wheel to avoid build issues
echo "=== Upgrading pip, setuptools, wheel ==="
pip install --upgrade pip setuptools wheel

# --------------------------
# MuJoCo 2.0 installation
# --------------------------
echo "=== Installing MuJoCo 2.0 ==="
mkdir -p ~/.mujoco

# Download from roboti.us if not already downloaded
MUJOCO_ZIP="$HOME/Downloads/mujoco200_linux.zip"
if [ ! -f "$MUJOCO_ZIP" ]; then
    wget https://roboti.us/download/mujoco200_linux.zip -O "$MUJOCO_ZIP"
fi

unzip -o "$MUJOCO_ZIP" -d ~/.mujoco

# --------------------------
# MuJoCo license key
# --------------------------
echo "=== Installing MuJoCo license key ==="
cat > ~/.mujoco/mjkey.txt << 'EOF'
MuJoCo Pro Individual license activation key, number 7777, type 6.

Issued to Everyone.

Expires October 18, 2031.

Do not modify this file. Its entire content, including the
plain text section, is used by the activation manager.

9aaedeefb37011a8a52361c736643665c7f60e796ff8ff70bb3f7a1d78e9a605
0453a3c853e4aa416e712d7e80cf799c6314ee5480ec6bd0f1ab51d1bb3c768f
8c06e7e572f411ecb25c3d6ef82cc20b00f672db88e6001b3dfdd3ab79e6c480
185d681811cfdaff640fb63295e391b05374edba90dd3e162a9d99b82a8b
ea3e87f2c67d08006c53daac2e563269cdb286838b168a2071c48c29fedfbea2
5effe96fe3cb05e85fb8af2d3851f385618ef8cdac42876831f095e052bd18c9
5dce57ff9c83670aad77e5a1f41444bec45e30e4e827f7bf9799b29f2c934e23
dcf6d3c3ee9c8dd2ed057317100cd21b4abbbf652d02bf72c3d322e0c55dcc24
EOF

# Add MuJoCo to library path immediately and permanently
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200_linux/bin"
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200_linux/bin" >> ~/.bashrc

# --------------------------
# Install Python packages
# --------------------------

# --------------------------
# Fix stable_baselines3 version.txt issue
# --------------------------
SB3_DIR="./stable_baselines3"
VERSION_FILE="$SB3_DIR/version.txt"

if [ ! -f "$VERSION_FILE" ]; then
    echo "0.11.0a1" > "$VERSION_FILE"
    echo "Created missing stable_baselines3/version.txt with 0.11.0a1"
fi

echo "=== Installing BPref main package ==="
pip install -e . 2>/dev/null

echo "=== Installing custom dm_control fork ==="
cd custom_dmcontrol
pip install -e . 2>/dev/null
cd ..

echo "=== Installing custom dmc2gym fork ==="
cd custom_dmc2gym
pip install -e . 2>/dev/null
cd ..

echo "=== Installing Metaworld (old version compatible) ==="
pip install "metaworld==0.0.1.dev0" 2>/dev/null # modify with link install

echo "=== Installing PyBullet ==="
pip install pybullet 2>/dev/null


# --------------------------
# Comments and instructions
# --------------------------
# - If you encounter OpenCV errors, see instructions in your original script about editing METADATA
# - PYTHONWARNINGS is NOT modified here, so Python will show warnings normally
# - LD_LIBRARY_PATH is set immediately, so MuJoCo will work in this shell session
# - You can now run training scripts like ./compare.sh or python train_SAC.py

echo "=== Setup complete! ==="
echo "Remember to run 'conda activate $ENV_NAME' before using BPref"

