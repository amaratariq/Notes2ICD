import pandas as pd
import numpy as np
import pickle as pkl
import sys

import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer

torch.manual_seed(0)

def generate_embedding(df):
    model_path = "./models/my-sup-simcse-clinicbert-az-ed-new"
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    print('CDE model loaded')
    sys.stdout.flush()

    cuda = torch.cuda.is_available()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(cuda, device)
    if cuda:
        model = model.to(device)
    sys.stdout.flush()
    k = 0
    keys= []
    for i in range(0,len(df)):
        sys.stdout.flush()
        texts = df.EXT_PRIM_DX.values[i].split('|')
        texts = [t for t in texts if t not in keys] #not already done
        if len(texts)>0:
            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

            if inputs['input_ids'].shape[1]>512:
                for k in inputs.keys():
                    inputs[k] = inputs[k][:, :512]

            with torch.no_grad():
                vec = model(**inputs.to(device), output_hidden_states=True, return_dict=True).pooler_output.cpu().detach().numpy()

            print(i)
            if i==0:
                mat = vec.reshape(-1, 768).copy()
            else:
                mat = np.concatenate((mat, vec.reshape(-1, 768)), axis=0)
            keys = keys+list(texts)           

    dct = {'mat': mat, 'keys':keys}
    return dct
