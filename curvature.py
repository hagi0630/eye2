# 曲率の分布計算
from matplotlib import pyplot as plt
import itertools
import cv2,random
import os, glob
import numpy as np
import pandas as pd
import labeling
import math
import pickle

# 近似円求める
def CircleFitting(x, y):
    """Circle Fitting with least squared
        input: point x-y positions
        output  cxe x center position
                cye y center position
                re  radius of circle
    """

    sumx = sum(x)
    sumy = sum(y)
    sumx2 = sum([ix ** 2 for ix in x])
    sumy2 = sum([iy ** 2 for iy in y])
    sumxy = sum([ix * iy for (ix, iy) in zip(x, y)])

    F = np.array([[sumx2, sumxy, sumx],
                  [sumxy, sumy2, sumy],
                  [sumx, sumy, len(x)]])

    G = np.array([[-sum([ix ** 3 + ix * iy ** 2 for (ix, iy) in zip(x, y)])],
                  [-sum([ix ** 2 * iy + iy ** 3 for (ix, iy) in zip(x, y)])],
                  [-sum([ix ** 2 + iy ** 2 for (ix, iy) in zip(x, y)])]])

    try:
        T = np.linalg.inv(F).dot(G)
    except np.linalg.LinAlgError:
        return 0, 0, float("inf")

    cxe = float(T[0] / -2)
    cye = float(T[1] / -2)

    try:
        re = math.sqrt(cxe ** 2 + cye ** 2 - float(T[2]))
    except :
        return cxe, cye, float("inf")
    # else:
    #     print(cxe ** 2 + cye ** 2)
    #     print(T[2])
    #     print(cxe ** 2 + cye ** 2 - float(T[2]))
    #     print(math.sqrt(cxe ** 2 + cye ** 2 - float(T[2])))
    #     exit()
    return cxe, cye, re

# 曲率のリストを返す
def calc_curvature_circle_fitting(x, y, npo=1):
    """
    Calc curvature
    x,y: x-y position list
    npo: the number of points using Calculation curvature
    ex) npo=1: using 3 point
        npo=2: using 5 point
        npo=3: using 7 point
    """

    cv = []
    n_data = len(x)

    for i in range(n_data):
        lind = i - npo
        hind = i + npo + 1

        if lind < 0:
            lind = 0
        if hind >= n_data:
            hind = n_data

        xs = x[lind:hind]
        ys = y[lind:hind]
        (cxe, cye, re) = CircleFitting(xs, ys)


        if re == float("inf"):
            cv.append(0.0)  # straight line
        else:
            cv.append(1/abs(re))

    return cv


labeling_img_list = []
for i in itertools.count():
    try:
        labeling_img_list.append(np.load(f"labeling_{i}.npy"))
    except:
        break

# with open("etval.pickle",mode="rb") as f:
#     etval_list = pickle.load(f)

with open("labels_dict.pickle",mode="rb") as f:
    labels_dict_list = pickle.load(f)

# 近似させる点の個数を決定。
# ex) npo=1: using 3 point
#  　　npo=2: using 5 point
#     npo=3: using 7 point
npo=3

for i,labeling_img in enumerate(labeling_img_list):
    curve=[]
    # etval = etval_list[i]
    labels_dict = labels_dict_list[i]
    for label,xy_list in labels_dict:
        label_x_list = []
        label_y_list = []
        for x,y in xy_list:
            label_x_list.append(x)
            label_y_list.append(y)
        if len(label_x_list)<=2*npo:
            continue
        curve+=calc_curvature_circle_fitting(label_x_list,label_y_list,npo=npo)

    plt.hist(curve)
    plt.savefig(f"curve_distribution_{i}.png")
    plt.clf()