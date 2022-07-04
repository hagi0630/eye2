# 分岐点削除。なお、分岐点を見つけるたびにそれを削除して画像更新する。こちらはあまり使わない。まとめて更新するのはdelete_branch2.py参照
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import os, glob
import numpy as np
import pandas as pd
# import area_compute

#分岐点かを確認する関数
def is_branch(img_skeleton,j,k):

    if img_skeleton[j][k]==255:
        #周囲8マス確認して3マス以上あるなら分岐と判定
        around_white = 0
        if img_skeleton[j][k+1]==255:
            around_white+=1
        if img_skeleton[j][k-1]==255:
            around_white+=1
        if img_skeleton[j+1][k+1]==255:
            around_white+=1
        if img_skeleton[j+1][k-1]==255:
            around_white+=1
        if img_skeleton[j+1][k]==255:
            around_white+=1
        if img_skeleton[j-1][k-1]==255:
            around_white+=1
        if img_skeleton[j-1][k+1]==255:
            around_white+=1
        if img_skeleton[j-1][k]==255:
            around_white+=1

        if around_white>=3:
            return True
        else:
            return False
    else:
        return False



files_skeleton = glob.glob("skeleton_GUOHALL_"+"*.png")
for i in range(len(files_skeleton)):

    img_skeleton = cv2.imread(files_skeleton[i],cv2.IMREAD_GRAYSCALE)

    for j in range(img_skeleton.shape[0]):
        for k in range(img_skeleton.shape[1]):
            if j==0 or k==0 or j==img_skeleton.shape[0]-1 or k==img_skeleton.shape[1]-1:
                continue
            elif is_branch(img_skeleton,j,k):
                img_skeleton[j][k]=0
            else:
                continue

    cv2.imwrite(f"guohall_delete_branch_{i}.png",img_skeleton)
