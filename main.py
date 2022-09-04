import pandas as pd
import numpy as np

import cv2
import glob
import time

import torch
import torchvision.models as models
import torchvision.transforms as transforms

from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

path_query = r'/Users/avimalhotra/Desktop/CS4186 assignment1/datasets_4186/query_4186'
path_query_text = r'/Users/avimalhotra/Desktop/CS4186 assignment1/datasets_4186/query_txt_4186'
path_gallery = r'/Users/avimalhotra/Desktop/CS4186 assignment1/datasets_4186/gallery_4186'

name_query = glob.glob(path_query + '/*.jpg')
text_query = glob.glob(path_query_text + '/*.txt')
num_query = len(name_query)


name_gallery = glob.glob(path_gallery + '/*.jpg')
num_gallery = len(name_gallery)


def similarity(query_feat, gallery_feat):
    return np.squeeze(cosine_similarity(query_feat, gallery_feat))


def query_crop(query_path, txt_path, save_path):
    query_img = cv2.imread(query_path)
    query_img = query_img[:, :, ::-1]

    txt = np.loadtxt(txt_path)
    crop = query_img[int(txt[1]):int(txt[1] + txt[3]), int(txt[0]):int(txt[0] + txt[2]), :]
    cv2.imwrite(save_path, crop[:, :, ::-1])
    return crop


def crop_all_queries():
    for n in range(num_query):
        query_crop(name_query[n], text_query[n], name_query[n])

def build_rank_list():
    pass



if "__main__" == __name__:
    crop_all_queries()

sift = cv2.ORB_create()
record_all = np.zeros((num_query, len(name_gallery)))

for i in range(2):
    time_s = time.time()
    dist_record = []

    per_query_name = name_query[i]
    per_query = cv2.imread(per_query_name)

    per_query_kp, per_query_des = sift.detectAndCompute(per_query, None)

    for j in range(num_gallery):
        per_gallery_name = name_gallery[j]
        per_gallery = cv2.imread(per_gallery_name)

        per_gallery_kp, per_gallery_des = sift.detectAndCompute(per_gallery, None)

        min_kp_num = np.amin([len(per_query_kp), len(per_gallery_kp)])
        query_part = per_query_des[:min_kp_num, :]
        gallery_part = per_gallery_des[:min_kp_num, :]

        dist_record.append(
            np.sum((np.double(query_part) - np.double(gallery_part)) ** 2) / np.prod(np.shape(query_part)))

    ascend_index = sorted(range(len(dist_record)), key=lambda k: dist_record[k])

    record_all[i, :] = ascend_index
    time_e = time.time()
    print('retrieval time for query {} is {}s'.format(i, time_e - time_s))

f = open(r'./rank_list.txt', 'w')
for i in range(num_query):
    f.write('Q' + str(i + 1) + ': ')
    for j in range(len(name_gallery)):
        f.write(str(np.int32(record_all[i, j])) + ' ')
    f.write('\n')
f.close()
