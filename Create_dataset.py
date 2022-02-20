#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 23:57:41 2021

@author: pushpa
"""

import pandas as pd
import numpy as np
import os
import json
from PIL import Image
from os import listdir
from os.path import isfile, join
os.environ['KMP_DUPLICATE_LIB_OK']='True'
path = '/home/pushpa/Documents/Doclm/dataset'
onlyfiles = [f for f in listdir(path+'/testing_data/annotations') if isfile(join(path+'/testing_data/annotations', f))]



def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    # resize image to 224x224
    image = image.resize((224, 224))
    image = np.asarray(image)  
    image = image[:, :, ::-1] # flip color channels from RGB to BGR
    image = image.transpose(2, 0, 1) # move channels to first dimension
    return image, (w, h)


def normalize_bbox(bbox, size):
    return [
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ]


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
    
    
    
def generate_examples(filepath):
        ann_dir = os.path.join(filepath, "annotations")
        img_dir = os.path.join(filepath, "images")
        for guid, file in enumerate(sorted(os.listdir(ann_dir))):
            tokens = []
            nbrs=[]
            bboxes = []
            ner_tags = []
            new_data=[]
            file_path = os.path.join(ann_dir, file)
            with open(file_path, "r", encoding="utf8") as f:
                data = json.load(f)
            image_path = os.path.join(img_dir, file)
            image_path = image_path.replace("json", "png")
            image, size = load_image(image_path)          
            
            for item in data["form"]:
                temp={}
                tokens=[]
                ner_tags=[]
                bboxes =[]
                nbrs=[]
                words, label = item["words"], item["label"]
                words = [w for w in words if w["text"].strip() != ""]
                if len(words) == 0:
                    continue
                if label == "other":
                    for w in words:
                        temp={}
                        temp['text'] =  w["text"]
                        temp['label'] ="O"
                        temp['box'] = normalize_bbox(w["box"], size)
                        new_data.append(temp)
                else:
                    temp={}
                    temp['text'] = words[0]["text"]
                    temp['label'] = "B-" + label.upper()
                    temp['box'] = normalize_bbox(words[0]["box"], size)
                    new_data.append(temp)
                    for w in words[1:]:
                        temp={}
                        temp['text'] = w["text"]
                        temp['label'] = "I-" + label.upper()
                        temp['box'] = normalize_bbox(w["box"], size)                                    
                        new_data.append(temp)
                
            f = open(filepath + '/changed/'+ file, "w")
            f.write(json.dumps(new_data,indent = 4, cls=NumpyEncoder))
            f.close()
                

generate_examples(path+'/training_data/')          
generate_examples(path+'/testing_data/')


