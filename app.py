import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Dimensionality Reduction Dashboard",
    layout="wide"
)

st.title("📉 Dimensionality Reduction Dashboard")
df = pd.read_csv("data/Iris.csv")

species = df["Species"]

X = df.drop(
    columns=["Id", "Species"],
    errors="ignore"
)

algorithm = st.selectbox(
    "Choose Technique",
    [
        "PCA",
        "t-SNE"
    ]
)

# PCA
if algorithm == "PCA":

    scaler = joblib.load(
        "models/scaler.pkl"
    )

    pca = joblib.load(
        "models/pca.pkl"
    )

    X_scaled = scaler.transform(X)

    X_reduced = pca.transform(X_scaled)

    st.subheader("Explained Variance Ratio")

    variance = pca.explained_variance_ratio_

    col1, col2 = st.columns(2)

    col1.metric(
        "PC1",
        round(variance[0], 3)
    )

    col2.metric(
        "PC2",
        round(variance[1], 3)
    )

# t-SNE
else:

    X_reduced = joblib.load(
        "models/tsne.pkl"
    )

# Plot

st.subheader(f"{algorithm} Visualization")

fig, ax = plt.subplots(figsize=(8,5))

scatter = ax.scatter(
    X_reduced[:,0],
    X_reduced[:,1]
)

ax.set_xlabel("Component 1")
ax.set_ylabel("Component 2")

ax.set_title(
    f"{algorithm} on Iris Dataset"
)

st.pyplot(fig)

# Dataset

reduced_df = pd.DataFrame(
    X_reduced,
    columns=[
        "Component_1",
        "Component_2"
    ]
)

reduced_df["Species"] = species

st.subheader("Reduced Dataset")

st.dataframe(
    reduced_df,
    use_container_width=True
)

# Download

csv = reduced_df.to_csv(
    index=False
)

st.download_button(
    "⬇ Download Reduced Dataset",
    csv,
    "reduced_dataset.csv",
    "text/csv"
)
