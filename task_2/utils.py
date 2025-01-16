import cv2
import torch
import kornia as K
import kornia.feature as KF
import matplotlib.pyplot as plt
from kornia_moons.feature import *
from kornia_moons.viz import draw_LAF_matches


class ImageMatcher:
    def __init__(self, device):
        self.device = device
        self.matcher = KF.LoFTR().eval().to(self.device)

    def match_images(self, img1, img2, conf=0.6):
        input = {            
            "image0": K.color.rgb_to_grayscale(img1),
            "image1": K.color.rgb_to_grayscale(img2),
        }

        with torch.inference_mode():
            mtch = self.matcher(input)

        conf_mask = torch.nonzero(mtch['confidence'] > conf)
        mkpts1 = mtch["keypoints0"][conf_mask].cpu().numpy()
        mkpts2 = mtch["keypoints1"][conf_mask].cpu().numpy()
        
        try:
            Fm, inliers = cv2.findFundamentalMat(mkpts1, mkpts2, cv2.USAC_MAGSAC, 0.8, 0.99, 10000)
            inliers = inliers > 0
        except:
            inliers = None
        
        return img1, img2, mkpts1, mkpts2, inliers

    def draw_laf_matches(self, img1, img2, mkpts1, mkpts2, inliers):
        if inliers is not None:
            graph = draw_LAF_matches(
                KF.laf_from_center_scale_ori(
                    torch.from_numpy(mkpts1).view(1, -1, 2),
                    torch.ones(mkpts1.shape[0]).view(1, -1, 1, 1),
                    torch.ones(mkpts1.shape[0]).view(1, -1, 1),
                ),
                KF.laf_from_center_scale_ori(
                    torch.from_numpy(mkpts2).view(1, -1, 2),
                    torch.ones(mkpts2.shape[0]).view(1, -1, 1, 1),
                    torch.ones(mkpts2.shape[0]).view(1, -1, 1),
                ),
                torch.arange(mkpts1.shape[0]).view(-1, 1).repeat(1, 2),
                K.tensor_to_image(img1),
                K.tensor_to_image(img2),
                inliers,
                draw_dict={"inlier_color": (0.2, 1, 0.2), "tentative_color": None, "feature_color": (0.2, 0.5, 1), "vertical": False},
            )
        else:
            fig, graph = plt.subplots(ncols=2, figsize=(18, 18))
            graph[0].imshow(K.tensor_to_image(img1))
            graph[1].imshow(K.tensor_to_image(img2))
        return graph