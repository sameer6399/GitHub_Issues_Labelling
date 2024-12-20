import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
import pyLDAvis.lda_model

import pickle

data = pd.read_csv('data/clustered_github_issues.csv')

data['cleaned_text'] = data['cleaned_text'].fillna('')

vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(data['cleaned_text'])

num_topics = 4
lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda.fit(X)

def print_top_words(model, feature_names, n_top_words=10):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic #{topic_idx}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))

with open('data/lda_model.pkl', 'wb') as f:
    pickle.dump(lda, f)

print("LDA model saved successfully!")

feature_names = vectorizer.get_feature_names_out()

print_top_words(lda, feature_names)

topic_dist = lda.transform(X)

lda_vis = pyLDAvis.lda_model.prepare(lda, X, vectorizer)

pyLDAvis.save_html(lda_vis, 'lda_visualization.html')

pyLDAvis.display(lda_vis)

topic_avg_dist = topic_dist.mean(axis=0)

plt.figure(figsize=(10, 6))
plt.bar(range(num_topics), topic_avg_dist, color='skyblue')

plt.title('Average Topic Distribution Across All Documents', fontsize=16)
plt.xlabel('Topic', fontsize=12)
plt.ylabel('Average Topic Probability', fontsize=12)

plt.xticks(range(num_topics), [f'Topic {i+1}' for i in range(num_topics)])
plt.savefig('bar_plot_topics.png', bbox_inches='tight')

plt.tight_layout()
plt.show()
