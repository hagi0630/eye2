# ラベリング処理
from collections import defaultdict
from matplotlib import pyplot as plt
import pickle
import cv2,random
import os, glob
import numpy as np
import pandas as pd

# ラベリングした画像に色塗り
def put_color_to_objects(src_img, label_table):
    label_img = np.zeros_like(src_img)
    for label in range(1,label_table.max() + 1):
        label_group_index = np.where(label_table == label)
        label_img[label_group_index] = random.sample(range(255), k=3)
    return label_img

files_delete_branch = glob.glob("test"+"*.png")
# files_delete_branch = glob.glob("guohall_delete_branch_"+"*.png")
labels_img_list = []
etval_list = []
labls_dict_list = []
for i in range(len(files_delete_branch)):
    img_delete_branch = cv2.imread(files_delete_branch[i],cv2.IMREAD_GRAYSCALE)
    src_img = cv2.imread(files_delete_branch[i])
    # ラベル付き画像を返す関数。etval:ラベル数、labels_img:ラベル番号が色の代わりについた、src_imgと同じ大きさの画像。
    etval, labels_img, stats, centroids = cv2.connectedComponentsWithStats(img_delete_branch)
    # cv2.imwrite("labels_test.png", put_color_to_objects(src_img, labels_img))
    labels_img_list.append(labels_img)
    etval_list.append(etval)

    # ラベル箇所をdictに保存。key:ラベル番号、value：位置のリスト
    label_dict = defaultdict(list)
    for j in range(labels_img.shape[0]):
        for k in range(labels_img.shape[1]):
            if labels_img[j][k]!=0:
                label_dict[labels_img[j][k]].append((j,k))
    # print(label_dict[1])
    label_dict = dict(sorted(label_dict.items()))
    labls_dict_list.append(label_dict)
    if i==0:
        break


for i in range(len(labels_img_list)):
    np.save(f"labeling_{i}.npy",labels_img_list[i])

with open("etval.pickle",mode="wb") as f:
    pickle.dump(etval_list,f)

with open("labels_dict.pickle",mode="wb") as f:
    pickle.dump(labls_dict_list,f)