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
    # 1. Determine which file to use (prioritize eval.csv)
    target_file = None
    if "eval.csv" in files:
        target_file = "eval.csv"
    elif "train.csv" in files:
        target_file = "train.csv"
    
    if not target_file:
        continue

    path = os.path.join(root, target_file)
    
    if os.path.getsize(path) == 0:
        continue

    # 2. Identify the label based on folder name
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
        # 3. Load and validate columns
        df = pd.read_csv(path, comment="#")
        if df.empty: continue

        # Flexible column selection
        reward_col = None
        for col in ["true_episode_reward", "episode_reward", "reward", "eval_reward"]:
            if col in df.columns:
                reward_col = col
                break
        
        if not reward_col or "step" not in df.columns:
            print(f"Skipping {path}: Required columns not found.")
            continue

        curves[label].append(df[reward_col].to_numpy())
        steps_dict[label] = df["step"].to_numpy()
        print(f"Loaded {label} from {target_file} in {folder}")

    except Exception as e:
        print(f"Error loading {path}: {e}")

if not curves:
    print("No valid data found. Check your folder names and CSV column headers.")
    exit()

# plot
plt.figure(figsize=(12, 7))

# 1. Pre-calculate and store data for sorting
plot_results = []

for label, runs in curves.items():
    min_len = min(len(r) for r in runs)
    data = np.vstack([r[:min_len] for r in runs])
    x = steps_dict[label][:min_len]
    
    mean = data.mean(axis=0)
    std = data.std(axis=0) if data.shape[0] > 1 else 0

    if SMOOTHING > 1:
        mean = pd.Series(mean).rolling(SMOOTHING, min_periods=1).mean()
        std = pd.Series(std).rolling(SMOOTHING, min_periods=1).mean()

    # Get the last valid value for sorting
    final_val = mean.iloc[-1]
    
    plot_results.append({
        "label": label,
        "x": x,
        "mean": mean,
        "std": std,
        "final_val": final_val,
        "num_runs": data.shape[0]
    })

# 2. Sort results by final_val in descending order
plot_results.sort(key=lambda x: x["final_val"], reverse=True)

# 3. Plot in the sorted order
for res in plot_results:
    color = COLOR_MAP.get(res["label"], "black")
    
    plt.plot(res["x"], res["mean"], 
             label=f"{res['label']} ({res['final_val']:.1f})", 
             color=color, lw=2)
    
    if res["num_runs"] > 1:
        plt.fill_between(res["x"], 
                         res["mean"] - res["std"], 
                         res["mean"] + res["std"], 
                         color=color, alpha=0.1)

plt.xlabel("Steps")
plt.ylabel("Reward")
plt.title("Performance Comparison (Ranked by Final Reward)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
