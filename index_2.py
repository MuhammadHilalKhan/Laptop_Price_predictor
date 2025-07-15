import streamlit as st
import numpy as np
import pandas as pa
import matplotlib.pyplot as plt
import base64
# in this function the above streamlit function occur genrate error so make sure in this function 
# don't write any function.
st.set_page_config(
    page_icon= "th_2.jpeg",
    page_title= "parictce"
)

# 1. Create Fstring: fstring is used for write multiple line of code 
# Convert local image to Base64
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Path to your image
image_path = "th.jpeg"  # Replace with your local image path
background_image = get_img_as_base64(image_path)
internal_css = f"""
<style>
.st-emotion-cache-1yiq2ps {{
    display: flex;
    flex-direction: row;
    -webkit-box-pack: start;
    place-content: flex-start;
    -webkit-box-align: stretch;
    align-items: stretch;
    position: absolute;
    inset: 0px;
    overflow: hidden;
}}


h1 {{
    font-family: "Source Sans Pro", sans-serif;
    font-weight: 700;
    color: rgb(169 43 13);
    padding: 1.25rem 0px 1rem;
    margin: 0px;
    line-height: 1.2;
}}

h3 {{
    font-family: "Source Sans Pro", sans-serif;
    font-weight: 600;
    color: rgb(42 245 60);
    letter-spacing: -0.005em;
    padding: 0.5rem 0px 1rem;
    margin: 0px;
    line-height: 1.2;
}}
.st-emotion-cache-12fmjuu {{
    position: fixed;
    top: 0px;
    left: 0px;
    right: 0px;
    height: 3.75rem;
    background: rgb(201 4 4);
    outline: none;
    z-index: 999990;
    display: block;
}}


</style>
"""
st.markdown(internal_css, unsafe_allow_html=True)
st.title("Wellcome to streamlit")

st.subheader("Hilal")
st.text("Hilal")
st.write("Hilal")

# text input

# date & times inputs

# selection inputs

# slider & select

# radio

# file upload 
