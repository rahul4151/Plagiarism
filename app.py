# app.py
#hello
import streamlit as st

# Import your different pages
from home import home_page
from contact import contact_page

# Define function to select page
def main():
    st.sidebar.title('Navigation')
    
    page = st.sidebar.selectbox('Go to', ['Home','Contact'])

    

    if page == 'Home':
        home_page()
    elif page == 'Contact':
        contact_page()

if __name__ == '__main__':
    main()













