import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
path = '/home/pushpa/Documents/Doclm/cord_original/cord_original'
import json
    
from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir(path+'/train/json') if isfile(join(path+'/train/json', f))]

for filename in onlyfiles:
    print(filename)
    annotationpath = path + "/train/json/"+ filename
    f = open(annotationpath, encoding="utf8")
    data = json.load(f)
    data = data['valid_line']
    newdata = []
    for item in data:
        for word in item['words']:
            temp = {}
            temp['text'] = word['text']
            temp['box'] = [word['quad']['x1'],word['quad']['y1'],word['quad']['x3'],word['quad']['y3']]
            temp['label'] = item['category']
            newdata.append(temp)
    f.close()
    f = open(path + '/train/changed/'+ filename, "w")
    f.write(json.dumps(newdata,indent = 4))
    f.close()
    