import streamlit as st
import pickle
import numpy as np

import streamlit as st

# Set page config (optional but recommended)
st.set_page_config(
    page_title="Your App Title",
    page_icon=":rocket:",
    layout="wide"
)

# Custom CSS with background and logo styling
st.markdown(
    """
    <style>
        /* Background image */
        .stApp {
            background-image: url("https://4kwallpapers.com/images/wallpapers/laptop-windows-11-3840x1080-10874.jpg");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        .st-emotion-cache-12fmjuu {
    position: fixed;
    top: 0px;
    left: 0px;
    right: 0px;
    height: 3.75rem;
    background: rgb(0, 0, 0);
    color: #fbfbfb;
    outline: none;
    z-index: 999990;
    display: block;
}
        /* Overlay for better readability */
        .stApp::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: rgba(255, 255, 255, 0.85);
            z-index: -1;
        }
        
        /* Main content container */
        .main .block-container {
            background-color: rgba(255, 255, 255, 0.9);
            border-radius: 10px;
            padding: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: rgba(255, 255, 255, 0.95);
            border-right: 1px solid #eee;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# import the model
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title("Laptop Predictor")

# brand

company = st.selectbox('Brand', df['Company'].unique())

# type of laptop

Type = st.selectbox('Type', df['TypeName'].unique())

# Ram

Ram = st.selectbox('RAM(in GB)', [2,4,6,8,12,16,24,32,64])

# Weight
weight = st.number_input('Weight of the laptop')
# Touchscreen
Touchscreen = st.selectbox('Touchscreen',['No','Yes'])
# IPS
ips = st.selectbox('IPS', ['No','Yes'])
# screen size
screen_size = st.number_input('Screen Size')
# resolution
resolution = st.selectbox('Screen Resolution', 
                          ['1024x600','1366x768','1440x900','1600x900',
                           '1920x1080','2560x1440','2560x1600','2880x1800',
                           '3200x1800','3840x2160'])


#  cpu
cpu = st.selectbox('CPU', df['Cpu brand'].unique())
# HDD
hdd = st.selectbox('HDD(in GB)', [0,128,256,512,1024,2048])
# SSD
ssd = st.selectbox('SSD(in GB)', [8,128,256,1024])
# GPU
gpu = st.selectbox('GPU',df['Gpu brand'].unique())
# OS
os = st.selectbox('OS', df['os'].unique())

if st.button("Predict Price"):
    # query
    ppi = None
    if Touchscreen == 'Yes':
        Touchscreen = 1
    else:
        Touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0
    try:
        x_res  =int(resolution.split('x')[0])
        y_res = int(resolution.split('x')[1])
        ppi = ((x_res**2) + (y_res**2))**0.5/screen_size
    except:
        ppi = 0
    query = np.array([company, Type, Ram, weight,Touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os ])

    query = query.reshape(1,12)
    st.title(int(np.exp(pipe.predict(query)[0])))



