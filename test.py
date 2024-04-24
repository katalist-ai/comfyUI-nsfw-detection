import numpy as np

from nudenet import NudenetDetector
import torch
from PIL import Image


def main():
    img = Image.open('nsfw_example.jpeg')
    arr = np.asarray(img)
    ten = torch.tensor(arr).unsqueeze(0)
    det = NudenetDetector()
    imgs = det.detect_and_blur(ten)
    img = imgs[0]
    img = img.squeeze(0).numpy()
    img = Image.fromarray(img)
    img.save('test_img_blurred.png')


if __name__ == '__main__':
    main()
