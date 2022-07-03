import cv2
import kornia as K
import kornia.feature as KF
import matplotlib.pyplot as plt
import os
import torch

from kornia_moons.feature import draw_LAF_matches

def draw_matching(image_dir, output_path, image_0_id, image_1_id, ext, mkpts0, mkpts1, inliers):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    image_1 = cv2.imread(os.path.join(image_dir, f"{image_0_id}.{ext}"))
    image_2 = cv2.imread(os.path.join(image_dir, f"{image_1_id}.{ext}"))
    image_1 = K.image_to_tensor(image_1, False).float() /255.
    image_2 = K.image_to_tensor(image_2, False).float() /255.
    image_1 = K.color.bgr_to_rgb(image_1).to(device)
    image_2 = K.color.bgr_to_rgb(image_2).to(device)

    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(1, 1, 1)
    draw_LAF_matches(
        KF.laf_from_center_scale_ori(torch.from_numpy(mkpts0).view(1,-1, 2),
                                    torch.ones(mkpts0.shape[0]).view(1,-1, 1, 1),
                                    torch.ones(mkpts0.shape[0]).view(1,-1, 1)),

        KF.laf_from_center_scale_ori(torch.from_numpy(mkpts1).view(1,-1, 2),
                                    torch.ones(mkpts1.shape[0]).view(1,-1, 1, 1),
                                    torch.ones(mkpts1.shape[0]).view(1,-1, 1)),
        torch.arange(mkpts0.shape[0]).view(-1,1).repeat(1,2),
        K.tensor_to_image(image_1),
        K.tensor_to_image(image_2),
        inliers,
        draw_dict={'inlier_color': (0.2, 1, 0.2),
                'tentative_color': None,
                'feature_color': (0.2, 0.5, 1),
                'vertical': False
                },
        ax=ax, # here
    )
    plt.savefig(output_path)
    plt.close()
    return