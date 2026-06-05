import pandas as pd
import joblib

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r"C:\Users\HARSHAVARDHAN\OneDrive\Desktop\TekWorks\phase2\5may\clustering_project\data\Iris.csv")

X = df.drop(
    columns=["Id", "Species"],
    errors="ignore"
)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

joblib.dump(
    scaler,
    "../models/scaler.pkl"
)

pca = PCA(n_components=2)

pca.fit(X_scaled)

joblib.dump(
    pca,
    "../models/pca.pkl"
)

print("PCA model saved successfully")