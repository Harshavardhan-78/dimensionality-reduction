import pandas as pd
import joblib

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r"C:\Users\HARSHAVARDHAN\OneDrive\Desktop\TekWorks\phase2\5may\clustering_project\data\Iris.csv")

X = df.drop(
    columns=["Id", "Species"],
    errors="ignore"
)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42
)

X_tsne = tsne.fit_transform(X_scaled)

joblib.dump(
    X_tsne,
    "../models/tsne.pkl"
)

print("t-SNE transformation saved successfully")