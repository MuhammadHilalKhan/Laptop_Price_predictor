"""
import streamlit as st
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt


# Step_1

# 1. Header() function:
st.header("This is a simple Header")
st.subheader('This is a subheader')
# 2. text() function: This function is used for simple text only.
st.text("This is a simple text ")
# 3. write() function: is used to display text, data, plots, and more in a Streamlit app.
#  It simplifies displaying various types of content by automatically adjusting to the input type.
st.write("we used a write function **text used for bold** and _textis used for italic_ ")

data = pd.read_csv("random_dataset.csv")
dc = { 'a': 76, 'b': 98, 'c': 10}
st.write(data) # OR also display a dataset like jupyter 
dc

# install matplotlib 
# fig, ax = plt.subplots()
# ax.scatter(np.arange(5), np.arange(5) ** 2)
# st.write(fig) 

# 5. If we can see a documentation of anythings through st.write function.
# st.write(st.write)
# st.write(st.error)

# 6. st.title() function.
st.title("This is the page Title")

# 7. code function is used only display a code not run. in the code function 
# pass the variable and spacified a language.
code = '''def func():
    print(np.arange(10))
'''
st.code(code, language='python')
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
# Step_2
# Data display Elements of streamlit.
#  1. Dataset Create and Display.
df = pd.DataFrame(
    np.random.randn(50, 20),
    columns=['cols' + str(i) for i in range(20)])
# st.write(df) 
# OR. The another way to display a dataframe & also manage the width & height.
st.dataframe(df, width=200, height=300)
# OR. Also dirctly used numpy file through dataframe & not include index & column names.
st.dataframe(np.random.randn(40, 10))
# 2. Tables.In table the intair thing are printed & No costumize & making it easy to understand & analyze.
st.table(df)

# 3. Matric.
st.metric('TCS stock', value = '3220.70', delta =  '19.10')
# OR
st.metric('TCS stock', value = '3220.70', delta =  '19.10', delta_color = 'inverse')
# OR
st.metric('TCS stock', value = '3220.70', delta =  '19.10', delta_color = 'off')

# 4. Json file. 
js = open('inter the path or dataset')
dl = json.load(js)
st.json(dl, expanded= True)
st.json(dl, expanded= False)
