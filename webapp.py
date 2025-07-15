import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
# in this function the above streamlit function occur genrate error so make sure in this function 
# don't write any function.
st.set_page_config(
    page_icon= "D:\Courses\Streamlit\image\powerpoint background 1.JPG",
    page_title= "WebApp"
)


st.title("@Dawar Company##")

# Encode the image in base64
with open("D:\Courses\Streamlit\image\powerpoint background 1.JPG", "rb") as img_file:
    base64_string = base64.b64encode(img_file.read()).decode()

page_bg = f"""
<style> 
.stApp {{
    background-image: url("data:image/jpeg;base64,{base64_string}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}}

h1 {{
    font-family: "Source Sans Pro", sans-serif;
    font-weight: 700;
    color: rgb(255 251 0);
    padding: 1.25rem 0px 1rem;
    margin: 0px;
    line-height: 1.2;
    background-color: black;
}}
.st-emotion-cache-12fmjuu {{
    position: fixed;
    top: 0px;
    left: 0px;
    right: 0px;
    height: 3.75rem;
    background: rgb(111, 255, 156);
    background-image: url("data:image/jpeg;base64,{base64_string}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    outline: none;
    z-index: 999990;
    display: block;
}}
p, ol, ul, dl {{
    margin: 0px 0px 1rem;
    padding: 0px;
    font-size: 1rem;
    font-weight: blod;
    color: black;
}}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# File upload section
st.sidebar.title("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load the data
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(df)

    # Display basic information
    st.write("### Basic Information")
    st.write(f"Number of rows: {df.shape[0]}")
    st.write(f"Number of columns: {df.shape[1]}")
    st.write("### Summary Statistics")
    st.write(df.describe())

    # Correlation Heatmap
    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Select column for visualization
    st.write("### Column Visualizations")
    column = st.selectbox("Select a column to visualize", df.columns)

    if pd.api.types.is_numeric_dtype(df[column]):
        # Histogram for numerical data
        st.write(f"### Histogram of {column}")
        fig, ax = plt.subplots()
        sns.histplot(df[column], kde=True, ax=ax, color="blue")
        sns.kdeplot(df[column], ax=ax, color="blue")
        st.pyplot(fig)
    else:
        # Bar chart for categorical data
        st.write(f"### Bar Chart of {column}")
        fig, ax = plt.subplots()
        df[column].value_counts().plot(kind="bar", ax=ax, color="orange")
        st.pyplot(fig)

else:
    st.write("Upload a CSV file to start EDA.")





