import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

BASE_DIR = "."
SMOOTHING = 5 
COLOR_MAP = {
    "SAC (Oracle)": "red", "PPO (Oracle)": "orange",
    "PrefPPO (2100)": "blue", "PrefPPO (1400)": "cyan", "PrefPPO (Pretrain 1400)": "purple",
    "PEBBLE (1400)": "green", "PEBBLE (700)": "olive", "PEBBLE (400)": "brown",
}

curves = defaultdict(list)
steps_dict = {}

print(f"Scanning {BASE_DIR} for data...")

for root, _, files in os.walk(BASE_DIR):
    if "train.csv" not in files:
        continue

    path = os.path.join(root, "train.csv")
    if os.path.getsize(path) == 0:
        continue

    folder = os.path.relpath(root, BASE_DIR).split(os.sep)[0]
    
    label = None
    if folder.startswith("oracle_sac"): label = "SAC (Oracle)"
    elif folder.startswith("oracle_ppo"): label = "PPO (Oracle)"
    elif folder.startswith("prefppo_pretrain_1400"): label = "PrefPPO (Pretrain 1400)"
    elif folder.startswith("prefppo_2100"): label = "PrefPPO (2100)"
    elif folder.startswith("prefppo_1400"): label = "PrefPPO (1400)"
    elif folder.startswith("pebble_1400"): label = "PEBBLE (1400)"
    elif folder.startswith("pebble_700"): label = "PEBBLE (700)"
    elif folder.startswith("pebble_400"): label = "PEBBLE (400)"

    if not label:
        continue

    try:
        df = pd.read_csv(path, comment="#")
        if df.empty: continue

        col = "true_episode_reward" if "true_episode_reward" in df.columns else "episode_reward"
        if col not in df.columns or "step" not in df.columns:
            continue

        curves[label].append(df[col].to_numpy())
        steps_dict[label] = df["step"].to_numpy()
        print(f"Loaded {label} from {path}")

    except Exception as e:
        print(f"Error loading {path}: {e}")

if not curves:
    print("No valid data found.")
    exit()

plt.figure(figsize=(10, 6))

for label, runs in curves.items():
    min_len = min(len(r) for r in runs)
    data = np.vstack([r[:min_len] for r in runs])
    x = steps_dict[label][:min_len]
    
    mean = data.mean(axis=0)
    std = data.std(axis=0) if data.shape[0] > 1 else 0

    if SMOOTHING > 1:
        mean = pd.Series(mean).rolling(SMOOTHING, min_periods=1).mean()
        std = pd.Series(std).rolling(SMOOTHING, min_periods=1).mean()

    color = COLOR_MAP.get(label, "black")
    plt.plot(x, mean, label=label, color=color, lw=2)
    if data.shape[0] > 1:
        plt.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)

plt.xlabel("Steps")
plt.ylabel("Reward")
plt.title("Performance Comparison")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()