"""
# My first app
Here's our first attempt at using data to create a table:
"""

import streamlit as st
import pandas as pd
import numpy as np

df = pd.DataFrame({
  'first column': [1, 2, 3, 4],
  'second column': [10, 20, 30, 40]
})

# CHANGE THEME
    # [theme]
    # base="light"
    # primaryColor="#5f5dff"
    
st.set_page_config(
    page_title="My first app",
    page_icon=":tada:",
    layout="centered"
)

add_selectbox = st.sidebar.selectbox(
    'How would you like to be contacted?',
    ('Email', 'Home phone', 'Mobile phone')
)

##### -----------
##### TEXTO Y TITULOS
##### -----------

st.title('My first app')
st.header('Header of my first app')
st.subheader('Subheader of my first app')
st.write("Here's our first attempt at using data to create a table:")
st.badge('This is a badge', icon='✅')

##### -----------
##### WIDGETS
##### -----------

players =st.slider('Select a number', 0, 100, 50)
st.write(f'You selected: {players} players')

if players > 50:
    st.success('You selected more than 50 players!')

if st.checkbox('Show dataframe'):
    st.dataframe(df)
    
# Insert a chat message container.
with st.chat_message("user"):
    st.write("Hello 👋")
    st.line_chart(np.random.randn(30, 3))

# # Display a chat input widget at the bottom of the app.
st.chat_input("Say something")
