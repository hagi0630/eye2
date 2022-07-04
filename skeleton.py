# 細線化処理
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import os, glob
import numpy as np
import pandas as pd

files = glob.glob("artery_"+"*.png")

for i,file in enumerate(files):
    img = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
    # 細線化(スケルトン化) THINNING_ZHANGSUEN
    skeleton1 = cv2.ximgproc.thinning(img, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    # 細線化(スケルトン化) THINNING_GUOHALL
    skeleton2 = cv2.ximgproc.thinning(img, thinningType=cv2.ximgproc.THINNING_GUOHALL)

    cv2.imwrite(f"skeleton_zhangsuen_{i}.png",skeleton1)
    cv2.imwrite(f"skeleton_GUOHALL_{i}.png",skeleton2)