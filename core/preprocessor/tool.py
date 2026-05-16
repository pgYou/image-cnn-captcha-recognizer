import numpy as np
import cv2 as cv
from PIL import Image


def Image2Mat(img):
    mat = np.asarray(img.capImg)
    mat = cv.cvtColor(mat, cv.COLOR_RGB2BGR)
    return mat

def Mat2Image(mat):
    img = Image.fromarray(cv.cvtColor(mat, cv.COLOR_BGR2RGB))
    return img
