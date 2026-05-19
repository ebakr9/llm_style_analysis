import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_excel(os.path.join(BASE_DIR, "..", "data", "responses.xlsx"))
results = {}

all_models = df["model"].unique()
all_ngrams = {}

for model_name in all_models:
    texts = df[df["model"] == model_name]["response"].tolist()
    vectorizer = CountVectorizer(
        ngram_range=(2, 2),
        max_features=50,
        stop_words="english"
    )
    X_ngram = vectorizer.fit_transform(texts).toarray()
    freq = X_ngram.sum(axis=0)
    ngram_freq = pd.Series(freq, index=vectorizer.get_feature_names_out())
    all_ngrams[model_name] = ngram_freq

# Tüm modelleri tek DataFrame'e topla
ngram_df = pd.DataFrame(all_ngrams).fillna(0)
print(ngram_df.shape)
print(ngram_df.head())


