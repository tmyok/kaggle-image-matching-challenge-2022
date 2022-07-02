import cv2
import os
import time

from exec_test_data import load_model_DKM, matching_DKM, seed_everything
from tools.draw_result import draw_matching

input_dir = "../input/image-matching-challenge-2022"
weight_dir = "../input/imc2022-pretrained-weights"
output_dir = "../output"
working_dir = "../output"

seed_everything(42)

image_dir = os.path.join(input_dir, "test_images", "d91db836")
image_1_id = "81dd07fb7b9a4e01996cee637f91ca1a"
image_2_id = "0006b1337a0347f49b4e651c035dfa0e"
ext = "png"

start = time.time()
DKM_param = {
    "model": load_model_DKM(os.path.join(weight_dir, "dkm"), working_dir),
    "scale": 1.5,
    "sample_num": 2000,
}
end = time.time()
print("load_model_DKM: ", end - start, " s")

start = time.time()
image_1 = cv2.imread(os.path.join(image_dir, f"{image_1_id}.{ext}"))
image_2 = cv2.imread(os.path.join(image_dir, f"{image_2_id}.{ext}"))
end = time.time()
print("cv2.imread: ", end - start, " s")
start = time.time()
_, mkpts1, mkpts2, _ = matching_DKM(image_1, image_2, DKM_param)
end = time.time()
print("matching_DKM: ", end - start, " s")

start = time.time()
_, inliers = cv2.findFundamentalMat(mkpts1, mkpts2, cv2.USAC_MAGSAC, 0.15, 0.9999, 20000)
inliers = inliers > 0
end = time.time()
print("cv2.findFundamentalMat: ", end - start, " s")

draw_matching(
    image_dir,
    os.path.join(output_dir, "sample_DKM.png"),
    image_1_id, image_2_id, ext,
    mkpts1, mkpts2, inliers)