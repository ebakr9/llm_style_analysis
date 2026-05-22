import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import os
from sklearn.model_selection import GridSearchCV
import shap
import pickle
import numpy as np
#Reading data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "..", "data", "sm_numeric_features.xlsx")
df = pd.read_excel(file_path)

#variables
X = df[[
    "avg_sentence_length",
    "punctuation_frequency", 
    "noun_ratio", "verb_ratio", "adj_ratio", "adv_ratio",
    "ST_TYPE_TOKEN_RATIO_LEMMAS",
    "G_PASSIVE", "G_ACTIVE",
    "G_PRESENT", "G_PAST",
    "L_I_PRON",
    "PS_CONTRADICTION", "PS_AGREEMENT"
]]
y = df["model"]

#RF model setup & cv
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#gridsearch parameters
param_grid = {
    "n_estimators": [5,10,20,30,50],
    "max_depth": [3, 5, 7, 10],
    "min_samples_leaf": [3, 5, 10]
}
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X, y)

results = pd.DataFrame(grid_search.cv_results_)
results = results[["params", "mean_test_score", "std_test_score"]].copy()

#train test accuracy for gridsearch params combinations
train_scores = []
for params in results["params"]:
    model = RandomForestClassifier(random_state=42, **params)
    model.fit(X, y)
    train_scores.append(model.score(X, y))

#choosing best params for both overfit and acc
results["train_accuracy"] = train_scores
results["overfit_gap"] = results["train_accuracy"] - results["mean_test_score"]
results["balanced_score"] = results["mean_test_score"] - 0.5 * results["overfit_gap"]
results = results.sort_values("balanced_score", ascending=False)
print(results.head(10).to_string(index=False))


# Final model w selected parameters
best_model = RandomForestClassifier(
    max_depth=3,
    min_samples_leaf=5,
    n_estimators=50,
    random_state=42
)

scores = cross_val_score(best_model, X, y, cv=cv, scoring="accuracy")
y_pred = cross_val_predict(best_model, X, y, cv=cv)

print(f"Accuracy: {scores.mean():.2f} (+/- {scores.std():.2f})")
print(classification_report(y, y_pred))

best_model.fit(X, y)
print(f"Train accuracy: {best_model.score(X, y):.2f}")
print(f"Test accuracy (CV): {scores.mean():.2f}")

#Shap values for attributes
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X)
np.save(os.path.join(BASE_DIR, "..", "data", "shap_values.npy"), shap_values)

#Confusion matrix save w pickle
with open(os.path.join(BASE_DIR, "..", "data", "classifier_results.pkl"), "wb") as f:
    pickle.dump({
        "y_true": y,
        "y_pred": y_pred,
        "category": df["category"].values,
        "feature_importance": pd.Series(best_model.feature_importances_, index=X.columns),
        "class_names": ["claude", "gemini", "gpt", "llama", "qwen"]
    }, f)

