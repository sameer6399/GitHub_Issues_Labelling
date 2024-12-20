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

df = pd.read_csv("data/github_issues.csv")

device = torch.device('mps') if torch.has_mps else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print(data.shape)

data = df.sample(n=250000, random_state=42)

"""# Data Preprocessing"""

data = data.drop_duplicates()
data = data.dropna(subset=['issue_title', 'body'])

data['combined_text'] = data['issue_title'].fillna('') + ' ' + data['body'].fillna('')

from langdetect import detect, LangDetectException

def is_english(text):
    try:
        language = detect(text)
        return language == 'en'
    except LangDetectException:
        return False
data = data[data['combined_text'].apply(is_english)]

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = ''.join(c if c.isalpha() or c.isspace() else ' ' for c in text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    text = ' '.join(words)
    return text

data['cleaned_text'] = data['combined_text'].apply(clean_text)

data.to_csv('data/preprocessed_data.csv', index=False)

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

embeddings = get_embeddings(data['cleaned_text'].tolist())

with open('data/embeddings.pkl', 'wb') as f:
    pickle.dump(embeddings, f)

print("Embeddings saved to embeddings.pkl")
