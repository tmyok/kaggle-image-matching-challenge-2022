import csv
import cv2
import numpy as np
import os
import pandas as pd

from collections import namedtuple
from tqdm import tqdm

Gt = namedtuple('Gt', ['K', 'R', 'T'])

def get_image_size(input_dir):

    df = pd.read_csv(os.path.join(input_dir, "scaling_factors.csv"))
    all_scene = df["scene"].unique()

    scene_list = []
    image_id_list = []
    width_list = []
    height_list = []
    for scene in all_scene:

        df = pd.read_csv(os.path.join(input_dir, scene, "calibration.csv"))
        image_ids = df["image_id"].values

        for image_id in tqdm(image_ids, desc=f"{scene}", leave=False, dynamic_ncols=True):
            img = cv2.imread(os.path.join(input_dir, scene, "images", f"{image_id}.jpg"))

            scene_list.append(scene)
            image_id_list.append(image_id)
            width_list.append(img.shape[1])
            height_list.append(img.shape[0])

    df_out = pd.DataFrame({
        "scene":scene_list,
        "image_id":image_id_list,
        "width":width_list,
        "height":height_list})

    return df_out

def get_rotation_info(input_csv):
    df = pd.read_csv(input_csv)
    rotation_matrices = df["rotation_matrix"].values
    image_ids = df["image_id"].values

    image_id_list = []
    x_list = []
    for image_id, rotation_matrix in zip(image_ids, rotation_matrices):
        image_id_list.append(image_id)
        rot = rotation_matrix.split(" ")
        x_list.append(rot[1])

    df_out = pd.DataFrame({"image_id":image_id_list, "x":np.array(x_list, dtype=np.float32)})

    return df_out

def LoadCalibration(filename):
    '''Load calibration data (ground truth) from the csv file.'''

    calib_dict = {}
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader):
            # Skip header.
            if i == 0:
                continue

            camera_id = row[0]
            K = np.array([float(v) for v in row[1].split(' ')]).reshape([3, 3])
            R = np.array([float(v) for v in row[2].split(' ')]).reshape([3, 3])
            T = np.array([float(v) for v in row[3].split(' ')])
            calib_dict[camera_id] = Gt(K=K, R=R, T=T)

    return calib_dict

if __name__ == "__main__":

    input_dir = "../../input/image-matching-challenge-2022/train/"
    output_dir = "../../output"

    # Load per-scene scaling factors.
    df_scale = pd.read_csv(os.path.join(input_dir, "scaling_factors.csv"))
    scene_list = df_scale["scene"].unique()

    # get validation_image_size
    print("get image size")
    df_validation_image_size = get_image_size(input_dir)

    # make validation csv
    print("make validation csv")
    for scene in scene_list:

        # Load scaling factor
        scale = df_scale[df_scale["scene"] == scene]["scaling_factor"].values[0]

        df = pd.read_csv(os.path.join(input_dir, scene, "pair_covisibility.csv"))

        # Load rotation info
        df_rotation = get_rotation_info(os.path.join(input_dir, scene, "calibration.csv"))

        # Load the ground truth.
        calib_dict = LoadCalibration(os.path.join(input_dir, scene, "calibration.csv"))

        df_validation_image_size_scene = df_validation_image_size[df_validation_image_size["scene"] == scene]
        image_width_dict = dict(zip(df_validation_image_size_scene["image_id"], df_validation_image_size_scene["width"]))
        image_height_dict = dict(zip(df_validation_image_size_scene["image_id"], df_validation_image_size_scene["height"]))

        image_pairs = df["pair"].values
        covisibility_image_pairs = df["covisibility"].values

        sample_ids = []
        batch_ids = []
        image_1_ids = []
        image_2_ids = []
        covisibility_list = []
        image_1_width_list = []
        image_1_height_list = []
        image_2_width_list = []
        image_2_height_list = []
        min_longest_edge_list = []
        image_1_rotation_list = []
        image_2_rotation_list = []
        dZ_list = []
        for pair, covisibility in tqdm(zip(image_pairs, covisibility_image_pairs), desc=f"{scene}", total=len(image_pairs), leave=False, dynamic_ncols=True):

            sample_id = f"phototourism;{scene};{pair}"
            image_1_id, image_2_id = pair.split("-")

            K1, R1_gt, T1_gt = calib_dict[image_1_id].K, calib_dict[image_1_id].R, calib_dict[image_1_id].T.reshape((3, 1))
            K2, R2_gt, T2_gt = calib_dict[image_2_id].K, calib_dict[image_2_id].R, calib_dict[image_2_id].T.reshape((3, 1))

            dR_gt = np.dot(R2_gt, R1_gt.T)
            dT_gt = (T2_gt - np.dot(dR_gt, T1_gt)).flatten()
            dZ = abs(dT_gt[2]) * scale

            image_1_width = image_width_dict[image_1_id]
            image_1_height = image_height_dict[image_1_id]
            image_2_width = image_width_dict[image_2_id]
            image_2_height = image_height_dict[image_2_id]
            min_longest_edge = min(max(image_1_width, image_1_height), max(image_2_width, image_2_height))

            image_1_rotation = df_rotation[df_rotation["image_id"] == image_1_id]["x"].values[0]
            image_2_rotation = df_rotation[df_rotation["image_id"] == image_2_id]["x"].values[0]

            sample_ids.append(sample_id)
            batch_ids.append(scene)
            image_1_ids.append(image_1_id)
            image_2_ids.append(image_2_id)
            covisibility_list.append(covisibility)
            image_1_width_list.append(image_1_width)
            image_1_height_list.append(image_1_height)
            image_2_width_list.append(image_2_width)
            image_2_height_list.append(image_2_height)
            min_longest_edge_list.append(min_longest_edge)
            image_1_rotation_list.append(image_1_rotation)
            image_2_rotation_list.append(image_2_rotation)
            dZ_list.append(dZ)

        df = pd.DataFrame({
            "sample_id":sample_ids,
            "batch_id":batch_ids,
            "image_1_id":image_1_ids,
            "image_2_id":image_2_ids,
            "covisibility":covisibility_list,
            "image_1_width":image_1_width_list,
            "image_1_height":image_1_height_list,
            "image_2_width":image_2_width_list,
            "image_2_height":image_2_height_list,
            "min_longest_edge":min_longest_edge_list,
            "image_1_rotation":image_1_rotation_list,
            "image_2_rotation":image_2_rotation_list,
            "dZ":dZ_list})
        df.to_csv(os.path.join(output_dir, f"{scene}.csv"), index=False)
