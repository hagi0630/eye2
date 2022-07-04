# 平均の太さ計算
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import os, glob
import numpy as np
import pandas as pd
import area_compute

files_skeleton = glob.glob("skeleton_zhangsuen_"+"*.png")
files_red = glob.glob("artery_"+"*.png")
for i in range(len(files_skeleton)):
    img = cv2.imread(files_red[i],cv2.IMREAD_GRAYSCALE)
    img_skeleton = cv2.imread(files_skeleton[i],cv2.IMREAD_GRAYSCALE)
    img_area = area_compute.area_compute(img)
    img_skeleton_area = area_compute.area_compute(img_skeleton)
    print(img_area/img_skeleton_area)