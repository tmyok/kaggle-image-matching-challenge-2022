import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

input_dir = "../../input/image-matching-challenge-2022/train"
output_dir = "../../output"

# Load per-scene scaling factors.
df = pd.read_csv(os.path.join(input_dir, "scaling_factors.csv"))
scene_list = df["scene"].unique()

for scene in tqdm(scene_list, dynamic_ncols=True):

    df = pd.read_csv(os.path.join(input_dir, scene, "calibration.csv"))

    rotation_matrix_list = df["rotation_matrix"].values

    x_list = []
    y_list = []
    for rotation_matrix in rotation_matrix_list:

        rot = rotation_matrix.split(" ")
        x_list.append(rot[1])
        y_list.append(rot[4])

    plt.scatter(np.array(x_list, dtype=np.float32), np.array(y_list, dtype=np.float32))
    plt.title(f"{scene}, n={len(x_list)}")
    plt.axis([-1,1,1.2,0])
    plt.savefig(os.path.join(output_dir, f"{scene}.png"))
    plt.close()