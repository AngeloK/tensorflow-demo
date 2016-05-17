#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from PIL import Image
from collections import OrderedDict

def image_to_array(img_file):
    img = Image.open(img_file).convert('LA')
    grayscaled_img = np.asarray(img.getdata(),dtype=np.float64)[:, 0] / 255.0
    reshaped_mat = grayscaled_img.reshape((1, grayscaled_img.shape[0]))
    return reshaped_mat

def create_pd_series(img, label):
    # Define columns for 20 * 20 image
    columns = [i for i in range(1, 401)]
    columns.append("class")
    img_mat = image_to_array(img)
    img_mat = img_mat.tolist()[0]
    img_mat.append(label)
    d = dict(zip(columns, img_mat))
    return OrderedDict(sorted(d.items()))

def preprocess_img():
    img_collection = []
    label = pd.read_csv("dataset/trainLabels.csv")
    index = []
    for i in range(1, 6284):
        img_collection.append(
            create_pd_series("dataset/trainResized/%d.Bmp" % int(i), label.ix[i-1, 1])
        )
        index.append(i)
    df = pd.DataFrame(data = img_collection, index=index)
    df.to_csv("train.csv")
