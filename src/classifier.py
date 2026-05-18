import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import os
from sklearn.model_selection import GridSearchCV

#Reading data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "..", "data", "numeric_features.xlsx")
df = pd.read_excel(file_path)

#variables
X = df[["avg_sentence_length", "lexical_diversity", "punctuation_frequency", 
        "discourse_marker_frequency", "noun_ratio", "verb_ratio", "adj_ratio", "adv_ratio"]]
y = df["model"]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#Random forest
clf_rf = RandomForestClassifier(random_state=42, n_estimators=20,max_depth=5,min_samples_leaf=3)
scores_rf = cross_val_score(clf_rf, X, y, cv=cv, scoring="accuracy")
y_pred_rf = cross_val_predict(clf_rf, X, y, cv=cv)

print(f"Random Forest - Accuracy: {scores_rf.mean():.2f} (+/- {scores_rf.std():.2f})")
print(classification_report(y, y_pred_rf))

# Overfitting kontrolü
clf_rf.fit(X, y)
print(f"RF Train accuracy: {clf_rf.score(X, y):.2f}")
print(f"RF Test accuracy (CV): {scores_rf.mean():.2f}")

# Feature importance
importances = pd.Series(clf_rf.feature_importances_, index=X.columns)
print(importances.sort_values(ascending=False))

# --- Logistic Regression ---
clf_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])
scores_lr = cross_val_score(clf_lr, X, y, cv=cv, scoring="accuracy")
y_pred_lr = cross_val_predict(clf_lr, X, y, cv=cv)

print(f"Logistic Regression - Accuracy: {scores_lr.mean():.2f} (+/- {scores_lr.std():.2f})")
print(classification_report(y, y_pred_lr))



param_grid = {
    "n_estimators": [5,10,20,35,50,70, 100, 200],
    "max_depth": [3, 5, 7, None],
    "min_samples_leaf": [1, 3, 5]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=cv,
    scoring="accuracy"
)

grid_search.fit(X, y)
print(f"Best params: {grid_search.best_params_}")
print(f"Best accuracy: {grid_search.best_score_:.2f}")

best_clf = grid_search.best_estimator_
train_accuracy = best_clf.score(X, y)
cv_accuracy = grid_search.best_score_

print(f"Train accuracy: {train_accuracy:.2f}")
print(f"CV accuracy: {cv_accuracy:.2f}")
print(f"Fark: {train_accuracy - cv_accuracy:.2f}")