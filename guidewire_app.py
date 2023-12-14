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
column_list = ["Cash and Cash Equivalents","Short-Term Investments","Net Receivables","Inventory","Other Current Assets","Total Current Assets","Long-Term Assets","Long-Term Investments","Fixed Assets","Goodwill","Intangible Assets","Other Assets","Deferred Asset Charges","Total Assets","Current Liabilities","Accounts Payable","Short-Term Debt / Current Portion of Long-Term Debt","Other Current Liabilities","Total Current Liabilities","Long-Term Debt","Other Liabilities","Deferred Liability Charges","Misc. Stocks","Minority Interest","Total Liabilities","Stock Holders Equity","Common Stocks","Capital Surplus","Retained Earnings","Treasury Stock","Other Equity","Total Equity","Total Liabilities & Equity","Net Income","Cash Flows-Operating Activities","Depreciation","Net Income Adjustments","Changes in Operating Activities","Accounts Receivable","Changes in Inventories","Other Operating Activities","Liabilities","Net Cash Flow-Operating","Cash Flows-Investing Activities","Capital Expenditures","Investments","Other Investing Activities","Net Cash Flows-Investing","Cash Flows-Financing Activities","Sale and Purchase of Stock","Net Borrowings","Other Financing Activities","Net Cash Flows-Financing","Effect of Exchange Rate","Net Cash Flow"," Total Revenue","Cost of Revenue","Gross Profit","Operating Expenses","Research and Development","Sales General and Admin","Operating Income","Add'l income/ expense items","Earnings Before Interest and Tax","Interest Expense","Earnings Before Tax","Income Tax","Minority Interest","Equity Earnings/Loss Unconsolidated Subsidiary","Net Income-Cont. Operations","Net Income","Net Income Applicable to Common Shareholders"]
cutoff = 20

# establish snowpark connection
conn = st.experimental_connection("snowpark")

# Reset the connection before using it if it isn't healthy
try:
    query_test = conn.query('select 1')
except:
    conn.reset()

# adding this to test out caching
st.cache_data(ttl=86400)

def plot_financials(df_2, x, y, x_cutoff, title):
    """"
    helper to plot the altair financial charts
   
    return st.altair_chart(alt.Chart(df_2.head(x_cutoff)).mark_bar().encode(
        x=x,
        y=y
        ).properties(title=title)
    ) 
  """ 
    df = pd.DataFrame(df_2)
    #st.write("Function-",df_2)
    #df_subset = df_2.head(x_cutoff)
  
    # Create a bar chart using st.bar_chart()

    return st.bar_chart(data=df,x=df.columns[0], y=df.columns[1:], color=None,width=0, height=300, use_container_width=True) 

    
def fs_chain(str_input):
    """
    performs qa capability for a question using sql vector db store
    the prompts.fs_chain is used but with caching
    """
    output = prompts.fs_chain(str_input)
    type(output)
    return output

# adding this to test out caching
st.cache_data(ttl=86400)
def sf_query(str_input):
    """
    performs snowflake query with caching
    """
    data = conn.query(str_input)
    return data

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
      selected = option_menu( menu_title="Menu",
      menu_icon = "search",
      options=['Finance Data', 'Annual Reports'], 
      icons=['database', 'filetype-pdf'],  
      default_index=0,
      styles={#"container":{"font-family": "Garamond"},
        "nav-link": {"font-family": "Source Sans Pro"},"font-size": "12px", "text-align": "left", "margin":"0px", "--hover-color": "grey"})
      #styles={"container":{"font-family": "Garamond"},
        #"nav-link": {"font-size": "20px", "text-align": "left", "margin":"0px", "--hover-color": "grey"}})
    if selected =='Finance Data':
        str_input = st.chat_input("Enter your question:")
        st.markdown("""
        AI assisted solution to extract information from your company's financial data like balance sheet, income statements, etc spanning across 2019 to 2022. Post your queries in the textbox at bottom of this page.
          
      
          **Few sample queries for reference are:**
      
          - What is the net income of Guidewire in 2022?
          - What is the gross profit in last 3 years?
        
        
        """)
        
        if "messages" not in st.session_state.keys():
              st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                role = message["role"]
                df_str = message["content"]
                if role == "user":
                    st.markdown(message["content"], unsafe_allow_html = True)
                    continue
                csv = StringIO(df_str)
                df_data = pd.read_csv(csv, sep=',')
                #st.write("Function2-",df_data)
                col1, col2 = st.columns(2)
                df_data.columns = df_data.columns.str.replace('_', ' ')
                headers = df_data.columns
                
                with col1:
                    st.markdown(tabulate(df_data, tablefmt="html",headers=headers,showindex=False), unsafe_allow_html = True)
                    #st.write(df_str)
                    
                    if len(df_data.index) >=2 & len(df_data.columns) == 2:
                        y = list(df_data.columns[1:])
                        title_name = df_data.columns[0]+'-'+df_data.columns[1]
                        with col2:
                                grph_ser_val_x1  = df_data.iloc[:,0]
                                grph_ser_val_y1  = df_data.iloc[:,1].apply(lambda x : float(x.replace(',','')))
                                frame = {df_data.columns[0] : grph_ser_val_x1,
                                         df_data.columns[1] : grph_ser_val_y1}
                                df_final1 = pd.DataFrame(frame)
                                plot_financials(df_final1,df_data.columns[0],df_data.columns[1], cutoff,title_name)

                            
        
        if prompt := str_input:
            st.chat_message("user").markdown(prompt, unsafe_allow_html = True)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            try:
                output = fs_chain(str_input)
                with st.expander("The SQL query used for above question is:"):
                  st.write(output['result'])
                try:
                    # if the output doesn't work we will try one additional attempt to fix it
                    query_result = sf_query(output['result'])
                    if len(query_result) >= 1:
                      with st.chat_message("assistant"):
                        df_2 = pd.DataFrame(query_result)
                        for name in df_2.columns:
                            if name in column_list:
                                new_name = f"{name} ($ thousands)"
                                df_2.rename(columns={name : new_name}, inplace=True)                          
                          
                        col1, col2 = st.columns(2)
                        df_2.columns = df_2.columns.str.replace('_', ' ')
                        headers = df_2.columns
                        with col1:
                         st.markdown(tabulate(df_2, tablefmt="html",headers=headers,showindex=False), unsafe_allow_html = True) 
                         
                        if len(df_2.index) >2 & len(df_2.columns) == 2:
                            y = list(df_2.columns[1:])
                            title_name = df_2.columns[0]+'-'+df_2.columns[1]
                            with col2:
                                grph_ser_val_x  = df_2.iloc[:,0]
                                grph_ser_val_y  = df_2.iloc[:,1].apply(lambda x : float(x.replace(',','')))
                                frame = {df_2.columns[0] : grph_ser_val_x,
                                         df_2.columns[1] : grph_ser_val_y}
                                df_final = pd.DataFrame(frame)
                                plot_financials(df_final, df_2.columns[0],df_2.columns[1], cutoff,title_name)
                               
                             #st.write(df_2)
                      #st.session_state.messages.append({"role": "assistant", "content": tabulate(df_2, tablefmt="html",headers=headers,showindex=False)})
                        st.session_state.messages.append({"role": "assistant", "content": df_2.to_csv(sep=',', index=False)})
                        
                    else:
                      with st.chat_message("assistant"):
                        st.write("Please try to improve your question. Note this tab is for financial statement questions. Use Tab 2 to ask from Annual Reports .")
      
                except Exception as error: 
                      #st.session_state.messages.append({"role": "assistant", "content": "The first attempt didn't pull what you were needing. Trying again..."})
                      output = fs_chain(f'You need to fix the code but ONLY produce SQL code output. If the question is complex, consider using one or more CTE. Examine the DDL statements and answer this question: {output}')
                      with st.expander("The SQL query used for above question is:"):
                          st.write(output['result'])
                      query_result = sf_query(output['result'])
                      df_2 = pd.DataFrame(query_result)
                      st.markdown(tabulate(df_2, tablefmt="html",headers=headers,showindex=False), unsafe_allow_html = True) 
                      
            except Exception as error:
              with st.chat_message("assistant"):
                st.markdown("Please try to improve your question. Note this tab is for financial statement questions. Use Tab 2 to ask from Annual Reports .")
                #st.session_state.messages.append({"role": "assistant", "content": "Please try to improve your question. Note this tab is for financial statement questions. Use Tab 2 to ask from Annual Reports ."})
              
    else:
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
