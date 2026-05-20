import streamlit as st
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Iris Flower Classification",
    page_icon="🌸",
    layout="centered"
)

# ---------- CUSTOM CSS ----------
st.markdown(
    """
    <style>
    h1 {
        color: #2E8B57;
        text-align: center;
    }
    h3 {
        color: #006400;
        text-align: center;
    }
    .result-box {
        background-color: #DFF6DD;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 20px;
        color: #2F4F4F;
        font-weight: bold;
    }
    .input-box {
        background-color: #F0F8FF;
        padding: 15px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- TITLE ----------
st.markdown("<h1>🌸 Iris Flower Classification</h1>", unsafe_allow_html=True)
st.markdown("<h3>Using Decision Tree Algorithm</h3>", unsafe_allow_html=True)
st.write("")

# ---------- LOAD DATA ----------
iris = load_iris()
X = iris.data
y = iris.target

# ---------- TRAIN MODEL ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(criterion="gini", random_state=42)
model.fit(X_train, y_train)

# ---------- INPUT FORM ----------
st.markdown("### 🌼 Enter Flower Measurements", unsafe_allow_html=True)
with st.container():
    st.markdown('<div class="input-box">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
        sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)

    with col2:
        petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
        petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- PREDICTION ----------
if st.button("🌸 Predict Flower"):
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(input_data)

    st.markdown(
        f'<div class="result-box">🌼 Predicted Flower: <br><br>{iris.target_names[prediction[0]]}</div>',
        unsafe_allow_html=True
    )

# ---------- FOOTER ----------
st.markdown(
    "<p style='text-align:center; color:#2F4F4F;'>Mini Project | Machine Learning | Decision Tree</p>",
    unsafe_allow_html=True
)
