# 面積計算
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import os, glob
import numpy as np
import pandas as pd

def area_compute(img):
    area=0
    print(img.shape)
    for j in range(img.shape[0]):
        for k in range(img.shape[1]):
            if img[j][k]>0:
                area+=1
    return area

files = glob.glob("artery_"+"*.png")

for i,file in enumerate(files):
    img = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
    print(area_compute(img))