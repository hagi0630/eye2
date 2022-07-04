from matplotlib import pyplot as plt
from PIL import Image
import cv2
import os, glob
import numpy as np
import pandas as pd

files = glob.glob("color_picture" + "/" + "*.png")

img_list = []
for i, file in enumerate(files):
    img = cv2.imread(file)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray_df = pd.DataFrame(np.array(img_gray))            # <class 'list'>
    # print(type(img_list))
    gray_df.to_csv(f'color_picture/gray{i+1}.csv')
    threshold = 1
    ret,img_twocolor = cv2.threshold(img_gray,threshold,255,cv2.THRESH_BINARY)
    two_df = pd.DataFrame(np.array(img_twocolor))            # <class 'list'>
    two_df.to_csv(f'color_picture/twocolor{i+1}.csv')
