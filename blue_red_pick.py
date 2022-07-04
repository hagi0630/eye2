from matplotlib import pyplot as plt
from PIL import Image
import cv2
import os, glob
import numpy as np
import pandas as pd

files = glob.glob("color_picture" + "/" + "*.png")
os.makedirs("artery", exist_ok=True)
os.makedirs("vein", exist_ok=True)

for i, file in enumerate(files):
    img = cv2.imread(file)
    blue_img,_,red_img = cv2.split(img)       #1色のみで塗られているのでこれで分割可能
    # print(blue_img.shape)
    cv2.imwrite(f"artery/red_{i}.png",red_img)
    cv2.imwrite(f"vein/blue_{i}.png",blue_img)
