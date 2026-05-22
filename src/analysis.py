# This script extracts the most common bigrams for each LLM and compares them.
# Bigrams (2-word phrases) capture word-pair patterns that single words miss,
# such as "for example" or "in order", which can reveal writing style differences.

import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_excel(os.path.join(BASE_DIR, "..", "data", "responses.xlsx"))

all_models = df["model"].unique()
all_ngrams = {}

for model_name in all_models:
    texts = df[df["model"] == model_name]["response"].tolist()
    vectorizer = CountVectorizer(
        ngram_range=(2, 2),   # bigrams only
        max_features=50,      # keep the 50 most frequent bigrams per model
        stop_words="english"
    )
    X_ngram = vectorizer.fit_transform(texts).toarray()
    freq = X_ngram.sum(axis=0)  # total count of each bigram across all texts
    ngram_freq = pd.Series(freq, index=vectorizer.get_feature_names_out())
    all_ngrams[model_name] = ngram_freq

# Combine into one DataFrame so bigram frequencies can be compared across models.
# Missing bigrams for a model are filled with 0.
ngram_df = pd.DataFrame(all_ngrams).fillna(0)
print(ngram_df.shape)
print(ngram_df.head())


