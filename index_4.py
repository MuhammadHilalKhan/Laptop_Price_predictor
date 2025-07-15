# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.express as px

# # Title and Description
# st.title("Exploratory Data Analysis (EDA) Tool")
# st.write("Upload your dataset to perform EDA interactively!")

# # File Upload
# uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
# if uploaded_file is not None:
#     # Load the data
#     df = pd.read_csv(uploaded_file)

#     # Display Dataset
#     st.subheader("Dataset Overview")
#     st.dataframe(df.head())

#     # Show Data Info
#     st.subheader("Basic Information")
#     st.write("Number of Rows:", df.shape[0])
#     st.write("Number of Columns:", df.shape[1])
#     st.write("Column Data Types:")
#     st.write(df.dtypes)
#     st.write("Missing Values:")
#     st.write(df.isnull().sum())

#     # Summary Statistics
#     st.subheader("Summary Statistics")
#     st.write(df.describe())

#     # Correlation Heatmap
#     st.subheader("Correlation Heatmap")
#     if st.checkbox("Show Correlation Heatmap"):
#         plt.figure(figsize=(10, 6))
#         sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
#         st.pyplot(plt)

#     # Interactive Visualizations
#     st.subheader("Interactive Visualizations")

#     # Numeric Column Selection for Distribution Plot
#     numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
#     if len(numeric_cols) > 0:
#         column = st.selectbox("Select Column for Distribution Plot", numeric_cols)
#         fig = px.histogram(df, x=column, nbins=30, title=f"Distribution of {column}")
#         st.plotly_chart(fig)
    
#     # Categorical Column Selection for Bar Chart
#     categorical_cols = df.select_dtypes(include=['object']).columns
#     if len(categorical_cols) > 0:
#         column = st.selectbox("Select Column for Bar Chart", categorical_cols)
#         fig = px.bar(df[column].value_counts(), title=f"Counts of {column}")
#         st.plotly_chart(fig)

#     # Pairplot
#     st.subheader("Pairplot")
#     if st.checkbox("Show Pairplot (Numeric Columns)"):
#         sns.pairplot(df[numeric_cols])
#         st.pyplot(plt)

#     # User-defined filtering
#     st.subheader("Filter Data")
#     column_filter = st.selectbox("Select Column to Filter", df.columns)
#     unique_values = df[column_filter].dropna().unique()
#     filter_value = st.selectbox(f"Select Value for {column_filter}", unique_values)
#     filtered_data = df[df[column_filter] == filter_value]
#     st.write(f"Filtered Data for {column_filter} = {filter_value}:")
#     st.dataframe(filtered_data)

# else:
#     st.write("Please upload a CSV file to start!")


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import base64

st.set_page_config(
    page_icon= "df.jpg",
    page_title= "parictce"
)

def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Path to your image
image_path = "df.jpg"  # Replace with your local image path
background_image = get_img_as_base64(image_path)

# Set Background Color using Custom CSS
page_bg = f"""
<style>
    .stApp {{
        background: url("https://tse2.mm.bing.net/th?id=OIP.8zJYriRG6Cboi60OoY2xpwHaEo&pid=Api&P=0&h=220");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    
    }}
    .main-title {{
        font-size: 32px;
        font-weight: bold;
        color: #4a4a8e;
    }}
    .section-title {{
        font-size: 24px;
        font-weight: bold;
        color: #2b2b52;
    }}
    .subsection {{
        margin-bottom: 25px;
    }}
#     .st-emotion-cache-12fmjuu {{
#     position: fixed;
#     top: 0px;
#     left: 0px;
#     right: 0px;
#     height: 3.75rem;
#     background: rgb(111 255 156);
#     outline: none;
#     z-index: 999990;
#     display: block;
# }}
.st-emotion-cache-12fmjuu {{
    position: fixed;
    top: 0px;
    left: 0px;
    right: 0px;
    height: 3.75rem;
    background: rgb(111 255 156);
    background-image: url("https://static.vecteezy.com/system/resources/previews/024/264/951/non_2x/blurred-spring-background-nature-with-blooming-glade-generative-ai-technology-free-photo.jpg"); /* Add your image URL here */
    background-size: cover; /* Ensure the image covers the area */
    background-position: center; /* Center the image */
    background-repeat: no-repeat; /* Prevent repeating */
    outline: none;
    z-index: 999990;
    display: block;
}}

</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# Title and Description
st.markdown("<h1 class='main-title'>Exploratory Data Analysis (EDA) Tool</h1>", unsafe_allow_html=True)
st.write("Upload your dataset to perform EDA interactively!")

# File Upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    # Load the data
    df = pd.read_csv(uploaded_file)

    # Use columns for layout
    with st.container():
        # Dataset Overview
        st.markdown("<h2 class='section-title'>Dataset Overview</h2>", unsafe_allow_html=True)
        st.dataframe(df.head(), use_container_width=True)

        # Basic Information
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())

        st.write("Column Data Types:")
        st.write(df.dtypes)

    # Summary Statistics
    st.markdown("<h2 class='section-title'>Summary Statistics</h2>", unsafe_allow_html=True)
    st.write(df.describe())

    # Correlation Heatmap
    st.markdown("<h2 class='section-title'>Correlation Heatmap</h2>", unsafe_allow_html=True)
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    st.pyplot(plt)

    # Interactive Visualizations
    st.markdown("<h2 class='section-title'>Interactive Visualizations</h2>", unsafe_allow_html=True)

    # Numeric Column Selection for Distribution Plot
    st.subheader("Distribution Plot")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 0:
        column = st.selectbox("Select Column for Distribution Plot", numeric_cols, key="numeric_dist")
        fig = px.histogram(df, x=column, nbins=30, title=f"Distribution of {column}")
        st.plotly_chart(fig)

    # Categorical Column Selection for Bar Chart
    st.subheader("Bar Chart")
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        column = st.selectbox("Select Column for Bar Chart", categorical_cols, key="categorical_bar")
        fig = px.bar(df[column].value_counts(), title=f"Counts of {column}")
        st.plotly_chart(fig)

    # Pairplot
    st.markdown("<h2 class='section-title'>Pairplot</h2>", unsafe_allow_html=True)
    if st.checkbox("Show Pairplot (Numeric Columns)"):
        sns.pairplot(df[numeric_cols])
        st.pyplot(plt)

    # User-defined filtering
    st.markdown("<h2 class='section-title'>Filter Data</h2>", unsafe_allow_html=True)
    column_filter = st.selectbox("Select Column to Filter", df.columns)
    unique_values = df[column_filter].dropna().unique()
    filter_value = st.selectbox(f"Select Value for {column_filter}", unique_values)
    filtered_data = df[df[column_filter] == filter_value]
    st.write(f"Filtered Data for {column_filter} = {filter_value}:")
    st.dataframe(filtered_data, use_container_width=True)

else:
    st.write("Please upload a CSV file to start!")
