import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# ----------------------------------
# App Title
# ----------------------------------
st.title("📰 News Article Clustering using K-Means")
st.write("Upload BBC News dataset and predict news category")

# ----------------------------------
# Upload CSV File
# ----------------------------------
uploaded_file = st.file_uploader("Upload BBC News CSV file", type=["csv"])

if uploaded_file is None:
    st.warning("Please upload the BBC News CSV file")
    st.stop()

# ----------------------------------
# Load Dataset
# ----------------------------------
df = pd.read_csv(uploaded_file)
st.success("Dataset loaded successfully!")

# ----------------------------------
# Verify 'descr' Column
# ----------------------------------
if "descr" not in df.columns:
    st.error("Dataset must contain a 'descr' column")
    st.write("Available columns:", df.columns.tolist())
    st.stop()

texts = df["descr"].astype(str)

# ----------------------------------
# TF-IDF Vectorization
# ----------------------------------
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.9,
    min_df=2
)

X = vectorizer.fit_transform(texts)

# ----------------------------------
# Train K-Means Model
# ----------------------------------
k = 5  # BBC dataset categories
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)

df["cluster"] = kmeans.labels_

# ----------------------------------
# Cluster → Topic Mapping (Manual)
# ----------------------------------
cluster_topic_map = {
    0: "Business",
    1: "Politics",
    2: "Sports",
    3: "Technology",
    4: "Entertainment"
}

# ----------------------------------
# Final Prediction
# ----------------------------------
st.subheader("🔮 Final Prediction")

user_article = st.text_area(
    "Enter a news article:",
    height=150
)

if st.button("Predict Category"):
    if user_article.strip() == "":
        st.warning("Please enter a news article")
    else:
        input_vec = vectorizer.transform([user_article])
        predicted_cluster = kmeans.predict(input_vec)[0]
        predicted_category = cluster_topic_map.get(predicted_cluster, "Unknown")

        st.success(f"Predicted Cluster: {predicted_cluster}")
        st.info(f"Predicted News Category: **{predicted_category}**")

# ----------------------------------
# Dataset Preview
# ----------------------------------
st.subheader("📂 Dataset Preview")
st.dataframe(df.head())
