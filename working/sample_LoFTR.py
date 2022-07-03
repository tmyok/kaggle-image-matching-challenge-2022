import cv2
import numpy as np
import os
import time
import torch

from exec_test_data import load_model_LoFTR, matching_loftr
from tools.draw_result import draw_matching

input_dir = "../input/image-matching-challenge-2022"
weight_dir = "../input/imc2022-pretrained-weights"
output_dir = "../output"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

image_dir = os.path.join(input_dir, "test_images", "d91db836")
image_1_id = "81dd07fb7b9a4e01996cee637f91ca1a"
image_2_id = "0006b1337a0347f49b4e651c035dfa0e"
ext = "png"

start = time.time()
LoFTR_param = {
    "device": device,
    "model": load_model_LoFTR(os.path.join(weight_dir, "kornia-loftr", "loftr_outdoor.ckpt"), device),
    "img_size": 840,
}
end = time.time()
print("load_model_LoFTR: ", end - start, " s")

start = time.time()
image_1 = cv2.imread(os.path.join(image_dir, f"{image_1_id}.{ext}"))
image_2 = cv2.imread(os.path.join(image_dir, f"{image_2_id}.{ext}"))
mask1 = np.ones((image_1.shape[1], image_1.shape[0]), dtype=np.uint8)
mask2 = np.ones((image_2.shape[1], image_2.shape[0]), dtype=np.uint8)
end = time.time()
print("cv2.imread: ", end - start, " s")

start = time.time()
mkpts1, mkpts2, _ = matching_loftr(image_1, image_2, mask1, mask2, LoFTR_param)
end = time.time()
print("matching_loftr: ", end - start, " s")

start = time.time()
_, inliers = cv2.findFundamentalMat(mkpts1, mkpts2, cv2.USAC_MAGSAC, 0.15, 0.9999, 20000)
inliers = inliers > 0
end = time.time()
print("cv2.findFundamentalMat: ", end - start, " s")

draw_matching(
    image_dir,
    os.path.join(output_dir, "sample_LoFTR.png"),
    image_1_id, image_2_id, ext,
    mkpts1, mkpts2, inliers)