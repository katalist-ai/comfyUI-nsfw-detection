import numpy as np

from nudenet import NudenetDetector
import torch
from PIL import Image


def main():
    img = Image.open('test_images/nsfw_example4.png')
    arr = np.asarray(img).astype(np.float32) / 255
    ten = torch.tensor(arr).unsqueeze(0)
    det = NudenetDetector()
    imgs = det.detect_and_blur(ten)
    img = imgs[0]
    img = img.squeeze(0).numpy() * 255
    img = Image.fromarray(img.astype(np.uint8))
    img.save('test_img_blurred.png')


if __name__ == '__main__':
    main()
