#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 19:51:19 2022

@author: pushpa
"""

#from transformers import LayoutLMv2Tokenizer, LayoutLMv2Model
import pandas as pd
import numpy as np
import os
import json
from PIL import Image
import csv
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from transformers import LayoutLMv2Tokenizer
from transformers import LayoutLMv2Processor
processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")
#tokenizer = LayoutLMv2Tokenizer.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")
path = '/home/pushpa/Documents/Doclm/SROIE2019-20210706T105403Z-001/archive/SROIE2019/dataset/'
from os import listdir
from os.path import isfile, join
padding_id = 0
def normalize_box(box, width, height):
     return [
         int(100 * (int(box[0]) / width)),
         int(100 * (int(box[1]) / height)),
         int(100 * (int(box[2]) / width)),
         int(100 * (int(box[3]) / height)),
     ]
def create_input_id(df,image_path):
    image = Image.open(image_path).convert("RGB")
    w,h = image.size
    #print(image_path)
    token_per_text=[]
    for index ,rows in df.iterrows():
        tokenid_list=[]
        for k in range(12):
                nbrs_index=rows['neighbours'][k]
                
                if nbrs_index==99999:
                    text=[" "]
                    box=[[0,0,0,0]]
                    word_labels=["O"]
                    encoding = processor(image ,text,boxes = box,word_labels=word_labels)
                    
                else:
                    #print(df.iloc[nbrs_index]["text"])
                    #print(df.iloc[nbrs_index]['box'])
                    #print(df.iloc[nbrs_index]['label'])
                    encoding = processor(image,[df.iloc[nbrs_index]["text"]],boxes = [df.iloc[nbrs_index]['box']],word_labels=[df.iloc[nbrs_index]['label']])
                tokens = encoding['input_ids']
                #print(tokens)
                if(len(tokens)>6):
                    tokens = tokens[:6]
                else:
                    tokens += [padding_id]*(6-len(tokens))
                tokenid_list.append(tokens)    
        token_per_text.append(tokenid_list)
    return token_per_text


def create_id_to_txt(df):
    token_per_text=[]
    for index ,rows in df.iterrows():
        tokenid_list=[]
        for k in range(12):
                nbrs_index=rows['neighbours'][k]
                
                if nbrs_index==99999:
                    text=""
                                        
                else:
                    text= df.iloc[nbrs_index]["text"]
                   
                tokenid_list.append(text)    
        token_per_text.append(tokenid_list)
    return token_per_text





def fetch_x0(lofl):
    return int(lofl[0])


def fetch_y0(lofl):
    return int(lofl[1])
def fetch_x1(lofl):
    return int(lofl[2])


def fetch_y1(lofl):
    return int(lofl[3])






def InRange(number,tol):
    #print("number",number)
    return tol >= abs(number) >= 0


def create_ver_nbrs(unique_x0,df,avg_diff_x0):
    nbrs_ver ={}
    for x0 in unique_x0:
        nbrs=[]
        current_x0 = x0
        for index ,rows in df.iterrows():
            tol =sum(avg_diff_x0)/len(avg_diff_x0)
            if InRange((rows['x0']-current_x0),tol):
                nbrs.append(index)
                #print(current_x0,rows["text"],df.iloc[index]["text"])
        nbrs_ver[x0]=nbrs
    return nbrs_ver
    
    
    
avg_font = 0    


def create_hz_nbrs(unique_y1,df,boundry_y):    
    nbrs_hz ={}
    global avg_font
    for y1 in unique_y1:
        current_y1 = y1
        #print(current_y1)
        nbrs=[]
        for index ,rows in df.iterrows():
            tol = abs(rows['y1']-current_y1)       
            if (tol< boundry_y):
                #print(rows['y1'])
                nbrs.append(index)
                #print(current_y1,rows["text"],df.iloc[index]["text"])
        nbrs_hz[y1]=nbrs
    return nbrs_hz
        
def remove_duplicates(x):
    x= set(x)
    return list(x)

def sort_nbrs(df):
    nbrs_per_text=[]
    for index ,rows in df.iterrows():
        
        #print(index)
        my_list =sorted(rows['neighbours'])
        result=[]
        if(my_list):
            start =len(my_list)//3 
            starter=my_list[start]
            #print("start",starter)
            result = my_list[my_list.index(starter):] + my_list[:my_list.index(starter)]
      
        nbrs_per_text.append(result)
    return nbrs_per_text

def trp(a):
    n =12
    diff=n-len(a)
    if diff >0:
         l= np.lib.pad(a,(0,diff),'constant', constant_values=99999)
         #l= list(map(lambda x: int(str(x).replace('99999', '')), l))
         return l
    else :
        return a[:n]


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
    
    
def create_boundry(df):
    count = 0
    sumx = 0
    sumy = 0
    for i ,row in df.iterrows():
        count+=1
        x1 = row['x0']
        y1 = row['y0']
        x2 = row['x1']
        y2 = row['y1']
        sumx += x2-x1
        sumy += y2-y1
    avgx = sumx//count
    avgx = avgx//2
    avgy = sumy//count
    avgy = avgy//2
    return avgx,avgy
    

#onlyfiles = [f for f in listdir(path+'/testing_data/annotations') if isfile(join(path+'/testing_data/annotations', f))]
nbrs_ver ={}
nbrs_hz ={}
#df=pd.DataFrame()
def generate_nbrs(filepath):
    global avg_font
    ann_dir = os.path.join(filepath, "changed")
    img_dir =  os.path.join(filepath, "images")
    for guid, file in enumerate(sorted(os.listdir(ann_dir))):
        file_path = os.path.join(ann_dir, file)
        img_path = os.path.join(img_dir, file)
        pre, ext = os.path.splitext(img_path)
        img_path = pre +'.jpg'
        images = Image.open(img_path).convert("RGB")
        w,h = images.size
        
        print(img_path)
        #global df
        df = pd.read_csv(str(file_path),sep='\t',names=["text","label","box"],quoting=csv.QUOTE_NONE, encoding='utf-8')
        #df = pd.read_json(str(file_path))
        df=df[df["text"].notna()].reset_index()
        df['box']=df['box'].astype(str)
        df['box']=df['box'].str.strip()
        df['box']=df['box'].str.split(" ")
        global nbrs_ver 
        global nbrs_hz 
        df['x0']=  df['box'].apply(fetch_x0).astype(int)
        df['y0']=  df['box'].apply(fetch_y0).astype(int)
        df['x1']=  df['box'].apply(fetch_x1).astype(int)
        df['y1']=  df['box'].apply(fetch_y1).astype(int)
        df['box'] = [(normalize_box(x,w,h)) for x in df['box']]
        #df=df.sort_values('y0')#.reset_index()#.drop("index",axis=1)
        unique_x0=df['x0'].unique()
        unique_x0=np.sort(unique_x0, axis = None)
        unique_y0=df['y0'].unique()
        unique_y0=np.sort(unique_y0, axis = None)
        unique_y1=df['y1'].unique()
        unique_y1=np.sort(unique_y1, axis = None)
        a = np.array(unique_y1)
        avg_diff_y1=np.diff(a)
        boundry_y=1.5*sum(avg_diff_y1)/len(avg_diff_y1)
        a = np.array(unique_x0)
        avg_diff_x0=np.diff(a)
        df["font_size"]=df['y1']-df['y0']
        df["font_diff"] = abs(df['font_size'] - df['font_size'].shift(-1))
        avg_font=df["font_size"].sum()/df.shape[0]
        avgx ,avgy=create_boundry(df)
        nbrs_ver=create_ver_nbrs(unique_x0,df,avg_diff_x0)
        nbrs_hz = create_hz_nbrs(unique_y1,df,boundry_y)
        b_series=[]
        for index ,rows in df.iterrows():
            #print("text",rows["text"])
            yindex=rows["y1"]
            to_append=nbrs_hz[yindex]
           
            to_append = [x for x in to_append if df.iloc[x]["text"]!=rows["text"]]
            #to_append = [x for x in to_append if len(str(df.iloc[x]["text"]))>2]
            #print("nbrs",to_append)
            b_series.append(to_append)
        df["nbrs_hz"]=b_series
            
            
        a_series=[]
        for index ,rows in df.iterrows():
            xindex=rows["x0"]
            #print(nbrs_ver[xindex])
            to_append = nbrs_ver[xindex]
            #print(index)
            to_append = [x for x in to_append if df.iloc[x]["text"]!=rows["text"]]
            #print(to_append)
            #to_append = [x for x in to_append if len(str(df.iloc[x]["text"]))>2]
            a_series.append(to_append)
        df["nbrs_ver"]=a_series
        
        
        df["neighbours"]=df['nbrs_hz'] #+df['nbrs_ver']
        #df["neighbours"]=df["neighbours"].apply(remove_duplicates)
        df['neighbours'] = pd.Series(sort_nbrs(df))
        df["neighbours"]=df["neighbours"].apply(trp)
        df['neighbours'] = pd.Series(create_input_id(df,img_path))
        #df['box'] =  df['box'].apply(lambda x: [int(i, 16) for i in x]) 
        #df['neighbors'] = pd.Series(create_id_to_txt(df))
        ##neighbors and neighbours are diff 
        df=df[['label', 'text', 'box','neighbours']]
        #df=df.sort_values('index').reset_index()
        #df=df.drop(["index","level_0"],axis=1)
        pre, ext = os.path.splitext(file_path)
        print(pre,ext)
       
        js=df.to_json(orient='records')
        with open(pre+".json", 'w') as f:
            f.write(js)
        os.remove(file_path)
           
        
        
        

generate_nbrs(path+'train')          
generate_nbrs(path+'test')
        
        
        
        
        
        
        
        
        
        
        
        
        
