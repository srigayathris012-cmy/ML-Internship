import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Page config
st.set_page_config(page_title="Spam SMS Detection", layout="centered")

st.title("📩 Spam SMS Detection using SVM")
st.write("Enter a message below to check whether it is **Spam** or **Ham**.")

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv("spam_sms.csv", encoding="latin-1")
    data = data[['v1','v2']]
    data.columns = ['Label','Message']
    data['Label'] = data['Label'].map({'ham':0,'spam':1})
    return data

data = load_data()

# Split data
X = data['Message']
y = data['Label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Vectorization
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

# Train SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_vec, y_train)

# User input
user_input = st.text_area("✍️ Enter SMS text:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        input_vec = vectorizer.transform([user_input])
        prediction = svm_model.predict(input_vec)[0]

        if prediction == 1:
            st.error("🚨 This message is **SPAM**")
        else:
            st.success("✅ This message is **HAM (Not Spam)**")
