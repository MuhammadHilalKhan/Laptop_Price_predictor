import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Streamlit App UI
st.title("🔬 Breast Cancer Prediction using KNN")

# Get feature input from user
st.subheader("Enter Feature Values:")

# Take first 5 features for simplicity (can be extended)
input_data = []
for i in range(5):
    val = st.slider(f"{data.feature_names[i]}", float(X[:, i].min()), float(X[:, i].max()))
    input_data.append(val)

# Fill remaining features with mean (to keep input simple)
mean_values = np.mean(X, axis=0)
input_data += list(mean_values[5:])

# Predict button
if st.button("Predict"):
    prediction = knn.predict([input_data])[0]
    result = data.target_names[prediction]
    st.success(f"The model predicts: **{result.upper()}**")

# Show accuracy
st.write("Model Accuracy:", knn.score(X_test, y_test))
