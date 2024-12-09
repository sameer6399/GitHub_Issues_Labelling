import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

import pickle

input_title = "to do ui"
input_body = "ui to do list - program icon - xml save file dialog - xml open file dialog"

combined_text = input_title + " " + input_body

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = ''.join(c if c.isalpha() or c.isspace() else ' ' for c in text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    text = ' '.join(words)
    return text

cleaned_text = clean_text(combined_text)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def get_embeddings(texts):
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).cpu().numpy())
    return np.vstack(embeddings)

embeddings = get_embeddings('cleaned_text'.split())

label_map = {0:"bug",1:"feature",2:"enhancement",3:"documentation"}

with open('data/kmeans_clusters_k_4.pkl', 'rb') as f:
    kmeans = pickle.load(f)

print(label_map[kmeans.predict(embeddings)[0]])

