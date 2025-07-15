import streamlit as st
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

df = pd.DataFrame(np.random.randn(10,2), columns=["prices", "diff"])
# 1. Line_chart.
st.line_chart(df)
st.line_chart(df, y= 'prices')

# 2. bar_chart.
st.bar_chart(df)
st.bar_chart(df, y='diff')

# 3. area_chart.
st.area_chart(df)
st.area_chart(df, y='prices')

# using Matplotlib
fig, ax = plt.subplots()
# ax.scatter(np.arange(10), np.arange(10) **2)
ax.hist(np.random.randn(100), bins=10)
st.pyplot(fig)
places = pd.DataFrame({
    "lat" : [34.95, 33.68],
    "lon" : [72.33, 73.04]
})
st.map(places)