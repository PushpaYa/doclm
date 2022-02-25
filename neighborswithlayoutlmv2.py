import pandas as pd
import os
dirpath="/home2/pushpa.y/dataset/"
dataset_path_train = "/home2/pushpa.y/dataset/training_data/images"
dataset_path_test = "/home2/pushpa.y/dataset/testing_data/images"
labels = ['O', 'B-HEADER', 'I-HEADER', 'B-QUESTION', 'I-QUESTION', 'B-ANSWER', 'I-ANSWER']
idx2label = {v: k for v, k in enumerate(labels)}
label2idx = {k: v for v, k in enumerate(labels)}
label2idx

def create_img_path(dataset_path):
    images_train = []
    dataset_fldr= dataset_path.split("/")[-2]
    for label_folder, _, file_names in os.walk(dataset_path):
      
      for _, _, image_names in os.walk(label_folder):
          #print(dataset_fldr)
          relative_image_names = []
          for image in image_names:
            relative_image_names.append( dirpath + "/"+dataset_fldr+'/images/'+ image)
          images_train.extend(relative_image_names)
          #print(relative_image_names)
    return images_train

image_path = create_img_path(dataset_path_train)
traindata_ = pd.DataFrame.from_dict({'image_path': image_path})




image_path = create_img_path(dataset_path_test)
testdata_ = pd.DataFrame.from_dict({'image_path': image_path})

#testdata_

import os
import json
from datasets import Dataset
from PIL import Image, ImageDraw, ImageFont
dirpath="/home2/pushpa.y/dataset/"
def generate_examples(example):
        path = example['image_path']
        image = Image.open(path)
        w,h = image.size
        filename = path.split('/')[-1]
        filename = filename.split('.')[0]
        datdir=path.split('/')[-3]
        #print(datdir)
        f = open(dirpath + '/' + datdir + '/changed/'+ filename + '.json')
        data = json.load(f)
        words = []
        boxes = []
        neighbors = []
        labels = []
        #print(data)
        for i in data:
          words.append(i['text'])
          unnormalized_box = i['box']
          normalized_box = unnormalized_box
          # if(i['text'] not in words_dictionary):
          #   global id
          #   words_dictionary[str(i['text'])] = id
          #   box_dictionary[id] = normalized_box
          #   id+=1
          boxes.append(normalized_box)
          neighbors.append(i['neighbors'])
          labels.append(label2idx[i['label']])
        # normalize the bounding boxes
        # add as extra columns 
        assert len(words) == len(boxes)
        example['words'] = words
        example['bbox'] = boxes

        example['neighbors'] = neighbors
        example['labels'] = labels
        return example

traindata = Dataset.from_pandas(traindata_)
train = traindata.map(generate_examples)
testdata = Dataset.from_pandas(testdata_)
test = testdata.map(generate_examples)



from PIL import Image
import numpy as np
from transformers import LayoutLMv2Processor
from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D
from torchvision.transforms import ToTensor
processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")



# we need to define custom features
features = Features({
    'image': Array3D(dtype="int64", shape=(3, 224, 224)),
    'labels': Sequence(feature=Value(dtype='int64')),
    'bbox': Array2D(dtype="int64", shape=(512, 4)),     
   
    'input_ids': Sequence(feature=Value(dtype='int64')),
    'attention_mask': Sequence(Value(dtype='int64')),
    'token_type_ids': Sequence(Value(dtype='int64')),
    'neighbors': Sequence(feature=Sequence(feature=Sequence(feature=Value(dtype='int64', id=None), length=-1, id=None), length=-1, id=None), length=-1, id=None)
    
})




def preprocess_data(examples):
  #print(examples['image_path'])
  path = examples['image_path']
  images = Image.open(path).convert("RGB")
  words = examples['words']
  bbox = examples['bbox']
  word_labels = examples['labels']
  
  encoded_inputs = processor(images, words, boxes=bbox, word_labels=word_labels,
                             padding="max_length", truncation=True)
  
   

  #print( len(encoded_inputs['image']))
  del examples['words']
  del examples['image_path']
  #print(len(encoded_inputs['input_ids']))
   # get required padding length
  pad_len = 512 - len(examples['neighbors'])
  token_neighbors = [[[0]*6]*12]*pad_len
  #print(token_neighbors)
  examples['neighbors'] = examples['neighbors'] + token_neighbors
  
  dict1 = {'neighbors':list(examples['neighbors'])}
  #dict1 = {'neighbors':examples['neighbors']}
  encoded_inputs.update(dict1)
  encoded_inputs["image"] = np.array(encoded_inputs["image"])
  



  
  
  return encoded_inputs


#test_dataset = test.map(preprocess_data, batched=True, remove_columns=test.column_names,
                                      #eatures=features)
#train_dataset =train.map(preprocess_data, batched=True, remove_columns=train.column_names,
                                      #features=features)




#encoded_dataset = train.map(lambda examples: preprocess_data(examples), batched=True)

import torch
train_dataset = []
for example in train:
    #print(example)
    processed_example = preprocess_data(example)
    example.update(processed_example)
    train_dataset.append(example)



test_dataset = []
for example in test:
    #print(example)
    processed_example = preprocess_data(example)
    example.update(processed_example)
    test_dataset.append(example)

#encoding['neighbors'][0][0]

encoding = train_dataset[0]
print(processor.tokenizer.decode(encoding['neighbors'][0][0]))




#encoding['neighbors']

from torch.utils.data.dataloader import default_collate
from torchvision.transforms import ToTensor
def collate_fn(batch):
   elem = batch[0]
   #print(type(elem))
   bbox = [item['bbox'] for item in batch]  # just form a list of tensor
   neighbors = [item['neighbors'] for item in batch]
   labels = [item['labels'] for item in batch]
   input_ids = [item['input_ids'] for item in batch]
   token_type_ids = [item['token_type_ids'] for item in batch]
   attention_mask = [item['attention_mask'] for item in batch]
   image =  [item['image'] for item in batch][0]
   elem= {'bbox' : torch.tensor(bbox), 'neighbors':torch.tensor(neighbors),'labels':torch.Tensor(labels),'input_ids':torch.Tensor(input_ids),'token_type_ids':torch.Tensor(token_type_ids),'attention_mask':torch.Tensor(attention_mask),'image':torch.Tensor(image)}
   #return dict(bbox = bbox, neighbors=neighbors,labels=labels,input_ids=input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask,image=image)
   #transposed = zip(*batch)
   #return [default_collate(samples) for samples in transposed]
   return elem

from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0,  collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=2,shuffle=True, num_workers=0,  collate_fn=collate_fn)



import torch
device = torch.device('cuda')



import torch.nn as nn
from transformers import LayoutLMv2Model
from transformers.models.layoutlm import LayoutLMConfig
from transformers.modeling_outputs import TokenClassifierOutput
import torchvision
from torchvision.ops import RoIAlign
import torch
import numpy as np
from transformers import LayoutLMv2Tokenizer
from torchnlp.nn import Attention
import time

class LayoutLMv2ForTokenClassification(nn.Module):
    def __init__(self): 
        super().__init__()
        self.start_time = time.time()
        # LayoutLM base model + token classifier
        self.num_labels = len(label2idx)
        self.layoutlm = LayoutLMv2Model.from_pretrained("microsoft/layoutlmv2-base-uncased", num_labels=self.num_labels)
        
        self.mlp = nn.Linear(13*768,12*768)
        self.sigmoid = nn.Sigmoid()
        self.attn = Attention(768)

        self.dropout = nn.Dropout(self.layoutlm.config.hidden_dropout_prob)
        self.classifier = nn.Linear(2*768, self.num_labels)

    def forward(self,image,input_ids,bbox,attention_mask,token_type_ids,neighbors,inputs_embeds=None,position_ids=None,head_mask=None,
        labels=None,output_attentions=None,output_hidden_states=None,return_dict=None,):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.

        """
        return_dict = return_dict if return_dict is not None else self.layoutlm.config.use_return_dict
        # print('layoutlm Start',time.time()-self.start_time)
        # first, forward pass on LayoutLM
        tokenizer = LayoutLMv2Tokenizer.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")
        outputs = self.layoutlm(
            image=image,
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds = inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # print('layoutlm End',time.time()-self.start_time)
        # print(self.layoutlm.config.hidden_size)
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        sequence_output = outputs[0][:, :seq_length]
        dictionary = {}
        batchsize = sequence_output.shape[0]

        final_neighbors = []
        # print('Making Neighbours',time.time()-self.start_time)
        for batch in range(batchsize):
          for i in range(512):
            dictionary[int(input_ids[batch][i])] = sequence_output[batch][i]
            
          l = []
          for i in range(512):
            # for n in neighbors[batch]:
            temp = []
            temp = torch.Tensor(temp).to(device)  
            for j in range(12):
              word_tokens = neighbors[batch][i][j]
              #print(word_tokens[0])
              embed = torch.zeros(768).to(device)
              for w in word_tokens:
                if(w.item() in dictionary):
                  x=w.cpu().detach().numpy()                  
                  x=int(x)                  
                  dictionary[x] =dictionary[x].cuda()
                  embed = torch.add(embed,dictionary[x])
                else:
                  embed = torch.add(embed,torch.zeros(768).to(device))
              temp = torch.cat((temp,embed),0)
              # print('temp',temp.shape)
              # temp = torch.stack(temp)
            l.append(temp)
          l = torch.stack(l)
          final_neighbors.append(l)
        final_neighbors = torch.stack(final_neighbors)
        # print('final_neighbors',final_neighbors.shape)
        neighbors = final_neighbors.to(device)
        # print('Neighbours Made',time.time()-self.start_time)
        # print(neighbors.shape)
        # print('Gating and Attention Start',time.time()-self.start_time)
        final_output = []
        batchsize = sequence_output.shape[0]
        for i in range(batchsize): 
          temp = []
          for row,n in zip(sequence_output[i],neighbors[i]):
            # print(row.shape,n.shape)
            rij = torch.cat((row,n),0)
            # print(row.shape)
            wr = self.mlp(rij)
            # print(wr.shape)
            g = self.sigmoid(wr)
            # print(g.shape)
            c_dash = torch.mul(g,n)
            # print(c_dash.shape)
            c_dash = torch.reshape(c_dash,(12,768))
            c_dash = c_dash.unsqueeze(0)
            row = row.unsqueeze(0)
            row = row.unsqueeze(0)
            # print(row.shape)
            # print(c_dash.shape)
            output,weights = self.attn(row,c_dash)
            temp.append(output[0][0])
          temp = torch.stack(temp)
          # temp = .unsqueeze(0)
          # print(final_output.shape)
          final_output.append(temp)
        final_output = torch.stack(final_output)
        final_output = torch.cat((sequence_output,final_output),2)
        # print('Gating And Attention End',time.time()-self.start_time)
        # print(final_output.shape)
        # print('Classification Start',time.time()-self.start_time)
        final_output = self.dropout(final_output)
        logits = self.classifier(final_output)
        # attention_mask = torch.cat((attention_mask,attention_mask),1)
        # labels = torch.cat((labels,labels),1)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        # print('Classification End',time.time()-self.start_time)
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



from transformers import AdamW
import torch
from tqdm.notebook import tqdm

from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler,DataLoader,TensorDataset
import pprint
import numpy
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

# k_folds = 10
results = []
# kfold = KFold(n_splits=k_folds, shuffle=False)
num_train_epochs = 45
t_total = len(train_dataloader) * num_train_epochs # total number of training steps 
torch.cuda.empty_cache()
model = LayoutLMv2ForTokenClassification()
#model = nn.DataParallel(model)
device = torch.device('cuda:0')
model = nn.DataParallel(model)
model.to(device)
optimizer = AdamW(model.parameters(), lr=2.373260750776288e-05)
# wandb.init(name='First Run', 
#           project='FinalResultsWithFullDatasetwith45epochs',
#           notes='FinalResultsWithFullDataset', 
#           tags=['FinalResultsWithFullDataset'])
global_step = 0
# wandb.watch(model)
model.train() 
for epoch in range(num_train_epochs):  
  print("Epoch:", epoch)
  for batch in tqdm(train_dataloader):
        # input_ids = torch.Tensor(list(batch['input_ids'].values)).to(device)
        input_ids = batch['input_ids'].long().cuda()
        image = batch['image'].long().cuda()
        bbox = batch["bbox"].long().cuda()
        attention_mask = batch["attention_mask"].long().cuda()
        token_type_ids = batch["token_type_ids"].long().cuda()
        labels = batch["labels"].long().cuda()
        neighbors = batch['neighbors'].long().cuda()
        # print(batch)
        # zero the parameter gradients
        optimizer.zero_grad()
        outputs = model(labels=labels,
                      input_ids= input_ids,
                      bbox=bbox, 
                      attention_mask=attention_mask,
                      token_type_ids=token_type_ids,
                      image=image,
                      neighbors= neighbors)
        # forward + backward + optimize
        # outputs = model(input_ids=input_ids,image=image,bbox=bbox,attention_mask=attention_mask,token) 
        loss = outputs.loss
        
        # print loss every 100 steps
        if global_step % 100 == 0:
          print(f"Loss after {global_step} steps: {loss.item()}")
        # wandb.log({
        # "Epoch": epoch,
        # "Train Loss": loss.item()})
        loss.backward()
        optimizer.step()
        global_step += 1

save_path = f'/home2/pushpa.y/final_model_45epochs.pth'
torch.save(model.state_dict(), save_path)

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import confusion_matrix,multilabel_confusion_matrix, classification_report
import numpy as np
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
eval_loss = 0.0
nb_eval_steps = 0
preds = None
out_label_ids = None
# testmodel = model
# # put model in evaluation mode
# testmodel.eval()
testmodel = LayoutLMv2ForTokenClassification()
testmodel.load_state_dict(torch.load(save_path))
testmodel.to(device)
for batch in tqdm(test_dataloader, desc="Evaluating"):
    with torch.no_grad():
        input_ids=batch['input_ids'].long().cuda()
        bbox=batch['bbox'].long().cuda()
        attention_mask=batch['attention_mask'].long().cuda()
        token_type_ids=batch['token_type_ids'].long().cuda()
        labels=batch['labels'].long().cuda()
        image = batch['image'].long().cuda()
        neighbors = batch['neighbors'].long().cuda()
        # resized_images = batch['resized_image'].to(device) 
        # resized_and_aligned_bounding_boxes = batch['resized_and_aligned_bounding_boxes'].to(device) 

        # forward pass
        outputs = testmodel(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids, 
                        labels=labels,image=image,neighbors=neighbors)

        # get the loss and logits
        tmp_eval_loss = outputs.loss
        logits = outputs.logits

        eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1

        # compute the predictions
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, labels.detach().cpu().numpy(), axis=0
            )

# compute average evaluation loss
eval_loss = eval_loss / nb_eval_steps
preds = np.argmax(preds, axis=2)

out_label_list = [[] for _ in range(out_label_ids.shape[0])]
preds_list = [[] for _ in range(out_label_ids.shape[0])]

for i in range(out_label_ids.shape[0]):
    for j in range(out_label_ids.shape[1]):
        if out_label_ids[i, j] != -100:
            out_label_list[i].append(idx2label[out_label_ids[i][j]])
            preds_list[i].append(idx2label[preds[i][j]])

testresults = {
    "loss": eval_loss,
    "precision": precision_score(out_label_list, preds_list),
    "recall": recall_score(out_label_list, preds_list),
    "f1": f1_score(out_label_list, preds_list),
    "classification_report": classification_report(out_label_list, preds_list)
    # "confusion_matrix": multilabel_confusion_matrix(MultiLabelBinarizer().fit_transform(out_label_list), MultiLabelBinarizer().fit_transform(preds_list))
    # "confusion_matrix": confusion_matrix(MultiLabelBinarizer().fit_transform(out_label_list), MultiLabelBinarizer().fit_transform(preds_list))
}
#print(out_label_list)
#print(preds_list)
print(testresults)

from datasets import load_metric
torch.cuda.empty_cache()
model = LayoutLMv2ForTokenClassification()
device = torch.device('cuda:0')
model = model.to(device)
metric = load_metric("seqeval")

# put model in evaluation mode
model.eval()
for batch in tqdm(test_dataloader, desc="Evaluating"):
    with torch.no_grad():
        input_ids=batch['input_ids'].long().cuda()
        bbox=batch['bbox'].long().cuda()
        attention_mask=batch['attention_mask'].long().cuda()
        token_type_ids=batch['token_type_ids'].long().cuda()
        labels=batch['labels'].long().cuda()
        image = batch['image'].long().cuda()
        neighbors = batch['neighbors'].long().cuda()

        # forward pass
        # forward pass
        outputs = testmodel(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids, 
                        labels=labels,image=image,neighbors=neighbors)
        
        # predictions
        predictions = outputs.logits.argmax(dim=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [idx2label[p.item()] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [idx2label[l.item()] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        metric.add_batch(predictions=true_predictions, references=true_labels)

final_score = metric.compute()
print(final_score)

labels_list = []
for i in out_label_list:
  labels_list.extend(i)
preds = []
for i in preds_list:
  preds.extend(i)
print(confusion_matrix(labels_list,preds,labels=['O', 'B-HEADER', 'I-HEADER', 'B-QUESTION', 'I-QUESTION', 'B-ANSWER', 'I-ANSWER']))

print(testresults["classification_report"])

