# Satellite image matching

The task is about matching two images by finding matching keypoints.

Finding suitable algorithm(neural network architecture) / gathering and preparing dataset / training
is very time consuming task. Hence to solve the task pretrained LoFTR model was used.

One of the main problems to solve is to figure out a way to preserve as much information
from satellite image as possible. In this project I used dumb way - just resized, but a lot
of research papers use their models on patches of images so image 10000x10000 would be split into 500x500 patches
matched and then stiched together to reproduce original image.


## How to install:

Runs with Python 3.11 (can be used with >3.11, but may require tweaks in requirements.txt)

Preferable way is to use poetry.

Otherwise use provided requirements.txt file.

```{bash}
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


## How to run:

1. Demo - demo.ipynb
2. Inference (same code as demo, with cli for image path input)
```{bash}
python inference.py -i img_path1 img_path2
```


## Ways to improve:

1. Use customized solution. Some of them:
    1. Better image matching model (https://www.sciencedirect.com/science/article/pii/S1569843223003989#da005)
    2. Transforming images (snow removal/cloud removal models)
2. Right now this project resizes images to specific resolution, better 'lossless' way would be to use the same algorithm on image patches. 
