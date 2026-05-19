import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "..", "data", "sm_numeric_features.xlsx")
df = pd.read_excel(file_path)

print(df.groupby("model")["PS_CONSEQUENCE"].agg(["mean", "std", lambda x: (x==0).sum()]))
print(df[["G_ACTIVE", "G_PASSIVE"]].corr())