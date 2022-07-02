import cv2
import numpy as np
import os
import torch

from exec_test_data import load_model_DKM, convert_DKM_image, seed_everything
from tools.draw_result import draw_matching

input_dir = "../input/image-matching-challenge-2022"
weight_dir = "../input/imc2022-pretrained-weights"
working_dir = "../output"
output_dir = "../output"

image_dir = os.path.join(input_dir, "test_images", "d91db836")
image_1_id = "81dd07fb7b9a4e01996cee637f91ca1a"
image_2_id = "0006b1337a0347f49b4e651c035dfa0e"
ext = "png"

image_path_1 = os.path.join(image_dir, f"{image_1_id}.{ext}")
image_path_2 = os.path.join(image_dir, f"{image_2_id}.{ext}")

seed_everything(42)

device = torch.device("cuda")
DKM_param = {
    "model": load_model_DKM(os.path.join(weight_dir, "dkm"), working_dir),
    "scale": 1.5,
    "sample_num": 2000,
}

input_image_1 = cv2.imread(image_path_1)
input_image_2 = cv2.imread(image_path_2)

matcher = DKM_param["model"]
scale = DKM_param["scale"]
sample_num = DKM_param["sample_num"]

_image_1 = convert_DKM_image(input_image_1)
_image_2 = convert_DKM_image(input_image_2)

# 推論を実行
dense_matches, dense_certainty = matcher.match(_image_1, _image_2)
dense_certainty = dense_certainty.sqrt()

# マッチング結果を取得
matches, confidence = (
    dense_matches.reshape(-1, 4).cpu().numpy(),
    dense_certainty.reshape(-1).cpu().numpy(),
)
mkpts1 = matches[:, :2]
mkpts2 = matches[:, 2:]

# 扱いやすいように座標変換
# Note that matches are produced in the normalized grid [-1, 1] x [-1, 1]
mkpts1[:, 0] = ((mkpts1[:, 0] + 1)/2)
mkpts1[:, 1] = ((mkpts1[:, 1] + 1)/2)
mkpts2[:, 0] = ((mkpts2[:, 0] + 1)/2)
mkpts2[:, 1] = ((mkpts2[:, 1] + 1)/2)

# DKM出力の描画
img1 = np.zeros((384, 512), dtype=np.uint8)
img2 = np.zeros((384, 512), dtype=np.uint8)
for pt1, pt2, conf in zip(mkpts1, mkpts2, confidence):
    img1[min(int(pt1[1]* 384), 383), min(int(pt1[0]*512), 511)] = int(conf*255)
    img2[min(int(pt2[1]* 384), 383), min(int(pt2[0]*512), 511)] = int(conf*255)

cv2.imwrite(os.path.join(output_dir,"01_DKM_output_1.png"), cv2.applyColorMap(img1, cv2.COLORMAP_JET))
cv2.imwrite(os.path.join(output_dir,"01_DKM_output_2.png"), cv2.applyColorMap(img2, cv2.COLORMAP_JET))

# 閾値を決める
relative_confidence = confidence/confidence.max()
relative_confidence_threshold = (cv2.threshold((relative_confidence*255.0).astype(np.uint8), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0]) / 255.0

# 閾値を超えたもののみを残す
idx = relative_confidence > relative_confidence_threshold
mkpts1 = mkpts1[idx]
mkpts2 = mkpts2[idx]
confidence = confidence[idx]

# フィルタンリングしたDKM出力の描画
img1 = np.zeros((384, 512), dtype=np.uint8)
img2 = np.zeros((384, 512), dtype=np.uint8)
for pt1, pt2, conf in zip(mkpts1, mkpts2, confidence):
    img1[min(int(pt1[1]* 384), 383), min(int(pt1[0]*512), 511)] = 255
    img2[min(int(pt2[1]* 384), 383), min(int(pt2[0]*512), 511)] = 255

cv2.imwrite(os.path.join(output_dir,"02_DKM_filter_1.png"), cv2.applyColorMap(img1, cv2.COLORMAP_BONE))
cv2.imwrite(os.path.join(output_dir,"02_DKM_filter_2.png"), cv2.applyColorMap(img2, cv2.COLORMAP_BONE))

# 外接矩形を計算
img_pair_rect = {
    "image1": [
        np.min(mkpts1[:, 0]), np.min(mkpts1[:, 1]),
        np.max(mkpts1[:, 0]), np.max(mkpts1[:, 1])],
    "image2": [
        np.min(mkpts2[:, 0]), np.min(mkpts2[:, 1]),
        np.max(mkpts2[:, 0]), np.max(mkpts2[:, 1])],
}

# 外接矩形の描画
img1 = np.zeros((384, 512), dtype=np.uint8)
img2 = np.zeros((384, 512), dtype=np.uint8)
for pt1, pt2, conf in zip(mkpts1, mkpts2, confidence):
    img1[min(int(pt1[1]* 384), 383), min(int(pt1[0]*512), 511)] = 255
    img2[min(int(pt2[1]* 384), 383), min(int(pt2[0]*512), 511)] = 255

# 外接矩形を描画
img1_xmin, img1_ymin, img1_xmax, img1_ymax = img_pair_rect["image1"]
img2_xmin, img2_ymin, img2_xmax, img2_ymax = img_pair_rect["image2"]
heatmap_img = cv2.applyColorMap(img1, cv2.COLORMAP_BONE)
cv2.rectangle(heatmap_img, (int(img1_xmin*512), int(img1_ymin*384)), (int(img1_xmax*512), int(img1_ymax*384)), (0, 255, 0), 2)
cv2.imwrite(os.path.join(output_dir,"03_heatmap_rect_1.png"), heatmap_img) # debug
heatmap_img = cv2.applyColorMap(img2, cv2.COLORMAP_BONE)
cv2.rectangle(heatmap_img, (int(img2_xmin*512), int(img2_ymin*384)), (int(img2_xmax*512), int(img2_ymax*384)), (0, 255, 0), 2)
cv2.imwrite(os.path.join(output_dir,"03_heatmap_rect_2.png"), heatmap_img) # debug

# 外接矩形を描画
temp_img = np.copy(input_image_1)
h, w = temp_img.shape[:2]
cv2.rectangle(temp_img, (int(img1_xmin*w), int(img1_ymin*h)), (int(img1_xmax*w), int(img1_ymax*h)), (0, 255, 0), 2)
cv2.imwrite(os.path.join(output_dir,"04_img_rect_1.png"), temp_img) # debug
temp_img = np.copy(input_image_2)
h, w = temp_img.shape[:2]
cv2.rectangle(temp_img, (int(img2_xmin*w), int(img2_ymin*h)), (int(img2_xmax*w), int(img2_ymax*h)), (0, 255, 0), 2)
cv2.imwrite(os.path.join(output_dir,"04_img_rect_2.png"), temp_img) # debug

# 対応点をサンプリング
sparse_matches, sparse_certainty = matcher.sample(
    dense_matches, dense_certainty,
    num = sample_num,
    relative_confidence_threshold = relative_confidence_threshold)

mkpts1 = sparse_matches[:, :2]
mkpts2 = sparse_matches[:, 2:]

h, w = input_image_1.shape[:2]
mkpts1[:, 0] = ((mkpts1[:, 0] + 1)/2) * w
mkpts1[:, 1] = ((mkpts1[:, 1] + 1)/2) * h

h, w = input_image_2.shape[:2]
mkpts2[:, 0] = ((mkpts2[:, 0] + 1)/2) * w
mkpts2[:, 1] = ((mkpts2[:, 1] + 1)/2) * h

confidence = sparse_certainty.reshape(-1)

_, inliers = cv2.findFundamentalMat(mkpts1, mkpts2, cv2.USAC_MAGSAC, 0.15, 0.9999, 10000)
inliers = inliers > 0

draw_matching(
    image_dir,
    os.path.join(output_dir, "sample_DKM.png"),
    image_1_id, image_2_id, ext,
    mkpts1, mkpts2, inliers)