import pandas as pd
import spacy
import os
from pathlib import Path
import stylo_metrix as sm

#Reading responses
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "..", "data", "responses.xlsx")
df_extractor = pd.read_excel(file_path)

nlp = spacy.load("en_core_web_sm")

#Sentence length without punctuation
def avg_sentence_length(text):
    doc = nlp(text)
    sentences = list(doc.sents)
    lengths = []
    for sent in sentences:
        word_count = 0
        for token in sent:
            if not token.is_punct:
                word_count += 1
        lengths.append(word_count)
    return sum(lengths) / len(lengths) if lengths else 0


#Punctuation frequency
def punctuation_frequency(text):
    doc = nlp(text)
    
    word_tokens = []
    for token in doc:
        if not token.is_punct:
            word_tokens.append(token)
    
    punct_tokens = []
    for token in doc:
        if token.is_punct:
            punct_tokens.append(token)
    
    return len(punct_tokens) / len(word_tokens) if word_tokens else 0



#POS distributions
def pos_distribution(text):
    doc = nlp(text)
    
    words = []
    for token in doc:
        if not token.is_punct:
            words.append(token)
    
    noun_count = 0
    verb_count = 0
    adj_count = 0
    adv_count = 0
    
    for token in words:
        if token.pos_ == "NOUN":
            noun_count += 1
        elif token.pos_ == "VERB":
            verb_count += 1
        elif token.pos_ == "ADJ":
            adj_count += 1
        elif token.pos_ == "ADV":
            adv_count += 1
    
    total = len(words)
    
    return {
        "noun_ratio": noun_count / total if total else 0,
        "verb_ratio": verb_count / total if total else 0,
        "adj_ratio": adj_count / total if total else 0,
        "adv_ratio": adv_count / total if total else 0
    }

#Stylometrix features
sm_nlp = sm.StyloMetrix("en")

#selected features
SM_FEATURES = ["ST_TYPE_TOKEN_RATIO_LEMMAS",
    "G_PASSIVE", "G_ACTIVE",
    "G_PRESENT", "G_PAST",
    "L_I_PRON", "L_WE_PRON",
    "PS_CONSEQUENCE", "PS_CONTRADICTION", "PS_AGREEMENT",
    "SY_NARRATIVE", "SY_QUESTION"
]

def get_stylometrix_features(text):
    result = sm_nlp.transform([text])
    return {feat: result[feat].values[0] for feat in SM_FEATURES}




#mergeing results
results = []

for index, row in df_extractor.iterrows():
    text = row["response"]
    
    pos = pos_distribution(text)
    sm_feats = get_stylometrix_features(text) 

    results.append({
        "prompt_id": row["prompt_id"],
        "category": row["category"],
        "model": row["model"],
        "avg_sentence_length": avg_sentence_length(text),
        "punctuation_frequency": punctuation_frequency(text),
        "noun_ratio": pos["noun_ratio"],
        "verb_ratio": pos["verb_ratio"],
        "adj_ratio": pos["adj_ratio"],
        "adv_ratio": pos["adv_ratio"],
        **sm_feats    
    })




#final results
df_features = pd.DataFrame(results)
output_path = os.path.join(BASE_DIR, "..", "data", "sm_numeric_features.xlsx")
df_features.to_excel(output_path, index=False)