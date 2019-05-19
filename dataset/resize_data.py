import numpy as np
from os import listdir
from PIL import Image
from os.path import join
from skimage.io import imread, imsave

DIR = 'milestone2_split/train/1k_images'
TARGET_H, TARGET_W = 100, 100


def dirty_resize(image):
    return np.array(Image.fromarray(image).resize((TARGET_H, TARGET_W)))

for filename in listdir(DIR):
    filepath = join(DIR, filename)

    orig_image = imread(filepath)
    resized_image = dirty_resize(orig_image)
    imsave(filepath, resized_image)
