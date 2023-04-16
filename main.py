import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import itertools
import math

if __name__ == "__main__":
    img = Image.open('./test_images/lena.tif')
    bw_image = np.asarray(img)
    print(bw_image.shape)