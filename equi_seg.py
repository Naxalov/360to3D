import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def show(arr):
    plt.imshow(arr)
    plt.show()


path_seg = os.path.join(os.getcwd(), 'image/equi_seg.png')
path_img = os.path.join(os.getcwd(), 'image/equirectangular.jpg')
seg = cv2.imread(path_seg)
img = cv2.imread(path_img)

H, W = seg.shape[:2]