# 太さの分布求める
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import os, glob
import numpy as np
import pandas as pd
from collections import deque,defaultdict

#ー２０，２０の正方形座標を考え、その格子点で中心から距離が短い順に、その距離（＝Key）に対応する座標（j,k）のリスト(=value)が入る
jk_distance=defaultdict(list)
for j in range(-20,20+1):
    for k in range(-20,20+1):
        jk_distance[(j**2+k**2)**(1/2)].append((j,k))

jk_distance = sorted(jk_distance.items(),key=lambda x:x[0])

#距離が小さい順に画素を見ていき、血管の外に達したらその直前の距離を血管の距離と定義
def distance(altery_img,first_j,first_k):
    pre_dis = 0.5
    for dis,jk_list in jk_distance:
        for j,k in jk_list:
            if altery_img[first_j+j][first_k+k]==0:
                return pre_dis
        pre_dis = dis



def thick_distribution(altery_img,skeleton_img):
    thick_list = []
    for j in range(skeleton_img.shape[0]):
        for k in range(skeleton_img.shape[1]):
            if skeleton_img[j][k] == 255:
                thick_list.append(distance(altery_img,j,k))
    return thick_list


altery_files = glob.glob("artery_"+"*.png")
skeleton_files = glob.glob("skeleton_GUOHALL_"+"*.png")
for i,file in enumerate(altery_files):
    artery_img = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
    skeleton_img = cv2.imread(skeleton_files[i],cv2.IMREAD_GRAYSCALE)
    thick_list = thick_distribution(artery_img,skeleton_img)
    plt.hist(thick_list,bins=20, range=(0,20))
    plt.savefig(f"thick_distribution_{i}.png")
    plt.clf()

