import torch
import argparse
import rasterio
import matplotlib.pyplot as plt
from utils import ImageMatcher
import torchvision.transforms as transforms
from rasterio.plot import reshape_as_image


# Global vars
RESIZE_TO = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tfs = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(RESIZE_TO),
])


def open_image(path):
    img = rasterio.open(path, driver="JP2OpenJPEG").read()
    img = reshape_as_image(img)
    img = tfs(img).unsqueeze(0).to(device)
    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script that detects mountain names in file"
    )
    parser.add_argument("-i", "--input", type=str, nargs="+", required=True, help="Input filename")
    args = parser.parse_args()
    files = args.input

    matcher = ImageMatcher(device)

    img1 = open_image(files[0])
    img2 = open_image(files[1])

    res = matcher.match_images(img1, img2)
    matcher.draw_laf_matches(*res)
    plt.show()

