# 分岐の角度の分布
from matplotlib import pyplot as plt
import itertools
import cv2,random
import os, glob
import numpy as np
import pandas as pd
import math
import pickle
from collections import defaultdict
import statistics
import warnings

warnings.filterwarnings('error')

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

#最小二乗誤差的に近似した近似直線の傾きを返す
def near_line(x,y):
    try:
        a,b = np.polyfit(x,y,1)
        return a
    except np.linalg.LinAlgError:
        return float("inf")

# 分岐箇所の角度を返す関数
def degree_compute(img, first_j, first_k, already_label_list):

    label_list = []
    nearlist_jk = []
    # 分岐点から見た相対的位置で、そのラベルがついた箇所がどこにあるか順に格納
    label_position = defaultdict(list)
    # Keyが中心からの距離、中心座標からの相対的位置がvalue
    jk_distance = defaultdict(list)
    # 分岐があると判定された箇所の周囲正方形[－5，5]を見てラベルを探す。分岐箇所で分岐と判定される箇所が複数画素合った場合周囲8マスだけでは見つけきれないため
    for j in range(-5,  5 + 1):
        for k in range(-5, 5 + 1):
            #ラベルが貼られているとき
            if img[first_j+j][first_k+k]!=0:
                label_list.append(img[first_j+j][first_k+k])
                jk_distance[(j ** 2 + k ** 2) ** (1 / 2)].append((j, k))

    # 距離が短い順に並べる
    jk_distance = dict(sorted(jk_distance.items(), key=lambda x: x[0]))
    # かぶり消し
    label_list = list(set(label_list))

    # ラベルが3つじゃないとき又は既に角度を求めたラベルセットのときは判定しない
    if len(set(label_list))!=3 or list(set(label_list)) in already_label_list:
        # print(0)
        return 0,already_label_list

    already_label_list.append(list(set(label_list)))

    # あるラベルのうち分岐点から最も近いものをnearlist_jkに記録
    for i in range(3):
        for jk_list in jk_distance.values():
            for j,k in jk_list:
                if img[first_j+j][first_k+k]==label_list[i]:
                    nearlist_jk.append((j,k))
                    break
            else:
                continue
            break

    #　あるラベルのうち分岐点から最も近い点が分かったのでそこから同じラベルを追って遠くにいき、分岐点からの相対的位置を記録
    # 10個記録し、10個もなければ0を返す
    for i in range(3):
        label = label_list[i]
        j=nearlist_jk[i][0]
        k=nearlist_jk[i][1]
        # 座標軸的にこの後は考えたいので、画像とは逆に上に行くほど大きくする
        label_position[label].append((j,k*(-1)))

        # 既に訪れた画素はラベル0にする
        img[j+first_j,k+first_k]=0

        for _ in range(9):
            # 8方向確認
            if img[j+first_j, k+first_k+1] == label:
                k = k+1
            elif img[j+first_j, k+first_k-1] == label:
                k = k-1
            elif img[j+first_j+1, k + first_k + 1] == label:
                j=j+1
                k=k+1
            elif img[j+1+first_j, k+first_k-1] == label:
                j=j+1
                k=k-1
            elif img[j+1+first_j, k+first_k] == label:
                j=j+1
            elif img[j-1+first_j, k + 1+first_k] == label:
                j=j-1
                k=k+1
            elif img[j-1+first_j, k - 1+first_k] == label:
                j=j-1
                k=k-1
            elif img[j-1+first_j, k+first_k] == label:
                j=j-1
            else:
                # print(_,j,k)
                return 0,already_label_list
            img[j + first_j, k + first_k] = 0
            label_position[label].append((j, k * (-1)))

    # Keyはラベル、valueは（near_lineで求めた傾き、平均のx座標、平均のy座標）
    label_ave_position = dict()
    for label,posi_list in label_position.items():
        sum_j, sum_k = 0, 0
        x_list = []
        y_list = []
        for j,k in posi_list:
            x_list.append(j)
            y_list.append(k)
            sum_j+=j
            sum_k+=k
        try:
            tilt = near_line(x_list,y_list)
        # 90度or-90度のとき
        except:
            tilt=float("inf")


        label_ave_position[label]=(tilt,sum_j/len(x_list),sum_k/len(y_list))

    # 各象限のリスト
    orthant2=[]
    orthant1=[]
    orthant4=[]
    orthant3=[]

    # その直線がどの象限に属するか
    for label,(tilt,j,k) in label_ave_position.items():
        if tilt==float("inf") and k>=0:
            orthant1.append((label,tilt,1))
        elif tilt==float("inf") and k<0:
            orthant3.append((label,tilt*(-1),3))
        elif j<0 and k>=0:
            orthant2.append((label,tilt,2))
        elif j>0 and k>=0:
            orthant1.append((label,tilt,1))
        elif j>0 and k<0:
            orthant4.append((label,tilt,4))
        elif j<0 and k<0:
            orthant3.append((label,tilt,3))

    # ラベルを次の順に並べたい。
    # （1）ｘ＝0より上にある。複数あるときはx軸の正の方向との角度が大きい順に
    # （2）ｘ＝0より下にある。複数あるときはx軸の正の方向との角度が小さい順に
    orthant1.sort(reverse=True,key=lambda x:x[1])
    orthant2.sort(reverse=True,key=lambda x:x[1])
    orthant3.sort(reverse=True,key=lambda x:x[1])
    orthant4.sort(reverse=True,key=lambda x:x[1])

    tilt_list = orthant2+orthant1+orthant4+orthant3

    # 並べた直線を（1）（2）（3）とすると、（1）（2）でなす角、（2）（3）でなす角、（3）（1）でなす角を求める
    # そのために、まずｘ軸の正の向きとの角度を求める。ただし、ｘ＝0より上にある場合はプラス、下にある場合はマイナスになる
    theta_list = []
    for label,tilt,orthant in tilt_list:
        if tilt==float("inf"):
            theta_list.append(90)
        elif tilt==(-1)*float("inf"):
            theta_list.append(-90)
        elif orthant==2:
            theta_list.append(math.degrees(math.atan2((-1)*tilt,-1)))
        elif orthant==1:
            theta_list.append(math.degrees(math.atan2(tilt,1)))
        elif orthant==4:
            theta_list.append(math.degrees(math.atan2(tilt,1)))
        else:
            theta_list.append(math.degrees(math.atan2((-1)*tilt, -1)))

    # 直線間の角度
    diff_theta = [theta_list[0]-theta_list[1],theta_list[1]-theta_list[2],360-(theta_list[0]-theta_list[1])-(theta_list[1]-theta_list[2])]
    # 一番小さい角度
    min_diff_theta = min(diff_theta)
    return min_diff_theta, already_label_list




files_skeleton = glob.glob("skeleton_GUOHALL_"+"*.png")
files_labeling_img = glob.glob("labeling_"+"*.npy")



for i in range(len(files_labeling_img)):
    # 角度格納
    degree_list = []

    img_skeleton = cv2.imread(files_skeleton[i],cv2.IMREAD_GRAYSCALE)
    img_labeling = np.load(files_labeling_img[i])

    # 既に確認したラベルセットを格納
    already_label_list = []

    for j in range(img_skeleton.shape[0]):
        for k in range(img_skeleton.shape[1]):
            if j==0 or k==0 or j==img_skeleton.shape[0]-1 or k==img_skeleton.shape[1]-1:
                continue
            elif is_branch(img_skeleton,j,k):
                degree,already_label_list = degree_compute(img_labeling,j,k,already_label_list)
                if degree>0:
                    degree_list.append(degree)

    plt.hist(degree_list)
    plt.savefig(f"degree_distribution_{i}.png")
    plt.clf()