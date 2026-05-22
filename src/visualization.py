import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#variables
df_visualization = pd.read_excel(os.path.join(BASE_DIR, "..", "data", "sm_numeric_features.xlsx"))
X = df_visualization[[
    "avg_sentence_length",
    "punctuation_frequency", 
    "noun_ratio", "verb_ratio", "adj_ratio", "adv_ratio",
    "ST_TYPE_TOKEN_RATIO_LEMMAS",
    "G_PASSIVE", "G_ACTIVE",
    "G_PRESENT", "G_PAST",
    "L_I_PRON",
    "PS_CONTRADICTION", "PS_AGREEMENT"
]]
class_names = ["claude", "gemini", "gpt", "llama", "qwen"]

#conf matrix and feature importance
with open(os.path.join(BASE_DIR, "..", "data", "classifier_results.pkl"), "rb") as f:
    results = pickle.load(f)

y_true = results["y_true"]
y_pred = results["y_pred"]
importances = results["feature_importance"]

corr_matrix = X.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()


#conf matrix
cm = confusion_matrix(y_true, y_pred, labels=class_names)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

#feature importance
importances_sorted = importances.sort_values(ascending=True)
plt.figure(figsize=(10, 6))
importances_sorted.plot(kind="barh")
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()


#SHAP values visualization
shap_values = np.load(os.path.join(BASE_DIR, "..", "data", "shap_values.npy"))

for i, model_name in enumerate(class_names):
    shap_vals_model = shap_values[:, :, i]
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals_model, X, plot_type="dot", show=False)
    plt.title(f"SHAP Summary — {model_name}", fontsize=14)
    plt.tight_layout()
    plt.show()


category = results["category"]

print("Accuracy by genre:\n")
for cat in pd.Series(category).unique():
    mask = pd.Series(category) == cat
    y_true_cat = pd.Series(y_true)[mask]
    y_pred_cat = pd.Series(y_pred)[mask]
    acc = (y_true_cat == y_pred_cat).mean()
    print(f"{cat}: {acc:.2f}")


#PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
colors = {"claude": "blue", "gemini": "green", "gpt": "red", "llama": "purple", "qwen": "orange"}

for model_name in df_visualization["model"].unique():
    mask = df_visualization["model"] == model_name
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                label=model_name, 
                color=colors[model_name],
                alpha=0.5, s=20)

plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.title("PCA — Stylistic Space of LLMs")
plt.legend()
plt.tight_layout()
plt.show()

#t-sne
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
for model_name in df_visualization["model"].unique():
    mask = df_visualization["model"] == model_name
    plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                label=model_name,
                color=colors[model_name],
                alpha=0.5, s=20)

plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title("t-SNE — Stylistic Space of LLMs")
plt.legend()
plt.tight_layout()
plt.show()

#tsne genre based
genre_colors = {
    "explanation": "red", "opinion": "blue", 
    "comparison": "green", "narration": "purple", 
    "argumentation": "orange", "dialogue": "brown"
}

plt.figure(figsize=(10, 8))
for cat in df_visualization["category"].unique():
    mask = df_visualization["category"] == cat
    plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                label=cat,
                color=genre_colors[cat],
                alpha=0.5, s=20)

plt.title("t-SNE — Colored by Genre")
plt.legend()
plt.tight_layout()
plt.show()