import snowflake.connector
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import prompts
from tabulate import tabulate
from PIL import Image
from streamlit_option_menu import option_menu
from io import StringIO

st.set_page_config(layout="wide")

username=st.secrets["streamlit_username"]
password=st.secrets["streamlit_password"]

# adding this to test out caching
st.cache_data(ttl=86400)

def creds_entered():
    if len(st.session_state["streamlit_username"])>0 and len(st.session_state["streamlit_password"])>0:
          if  st.session_state["streamlit_username"].strip() != username or st.session_state["streamlit_password"].strip() != password: 
              st.session_state["authenticated"] = False
              st.error("Invalid Username/Password ")

          elif st.session_state["streamlit_username"].strip() == username and st.session_state["streamlit_password"].strip() == password:
              st.session_state["authenticated"] = True


def authenticate_user():
      if "authenticated" not in st.session_state:
        buff, col, buff2 = st.columns([1,1,1])
        col.text_input(label="Username:", value="", key="streamlit_username", on_change=creds_entered) 
        col.text_input(label="Password", value="", key="streamlit_password", type="password", on_change=creds_entered)
        return False
      else:
           if st.session_state["authenticated"]: 
                return True
           else:  
                  buff, col, buff2 = st.columns([1,1,1])
                  col.text_input(label="Username:", value="", key="streamlit_username", on_change=creds_entered) 
                  col.text_input(label="Password:", value="", key="streamlit_password", type="password", on_change=creds_entered)
                  return False

if authenticate_user():
    with st.sidebar:
      image = Image.open("assets/jadefinance.png")
      image = st.image('assets/jadefinance.png',width=290)

    query = st.chat_input("Enter your question:")
    st.markdown("""

    AI assisted solution to extract information from your company's annual reports from 2019 to 2022. Post your queries in the textbox at bottom of this page.
          
    **Few sample queries for reference are:**
  
    - What are the operating expenses of the Guidewire for last 2 years?
    - What are the risks Guidewire is facing?
    
    """)
    
    # Create a text input to edit the selected question
    if "messages_1" not in st.session_state.keys():
          st.session_state.messages_1 = []

    for message in st.session_state.messages_1:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html = True)
    
    if prompt1 := query:
        st.chat_message("user").markdown(prompt1, unsafe_allow_html = True)
          # Add user message to chat history
        st.session_state.messages_1.append({"role": "user", "content": prompt1})
        try:
            with st.chat_message("assistant"):
              result = prompts.letter_chain(query)
              #st.write(result['result'])
              answer = result['result']
              st.markdown(f'<p style="font-family:sans-serif; font-size:15px">{answer}</p>', unsafe_allow_html=True)
              st.session_state.messages_1.append({"role": "assistant", "content":answer } )

        except:
            st.write("Please try to improve your question")
