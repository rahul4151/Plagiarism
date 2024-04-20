# home.py
import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from PyPDF2 import PdfReader
from nltk.corpus import stopwords
import string

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join([word for word in text.split() if word not in stopwords.words("english")])
    return text

def home_page():
    st.title("Plagiarism Detection")
    
    # Create a form using st.form()
    uploaded_file1 = st.file_uploader("Upload PDF file 1", type="pdf")
    uploaded_file2 = st.file_uploader("Upload PDF file 2", type="pdf")

    # Display the uploaded PDFs side by side
    if uploaded_file1 is not None and uploaded_file2 is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.write("You have uploaded the following PDF file 1:")
            

        with col2:
            st.write("You have uploaded the following PDF file 2:")
            

        # Submit button
        if st.button("Submit"):
            st.write("Submitted!")

            with st.spinner('Extracting text from PDF files...'):
                pdf_reader1 = PdfReader(uploaded_file1)
                text1 = ''
                for page in pdf_reader1.pages:
                    text1 += page.extract_text()
                
                pdf_reader2 = PdfReader(uploaded_file2)
                text2 = ''
                for page in pdf_reader2.pages:
                    text2 += page.extract_text()

    # Preprocess text
            text1 = preprocess_text(text1)
            text2 = preprocess_text(text2)

            cv = CountVectorizer()
            vectors = cv.fit_transform([text1,text2])

            features = pd.DataFrame(vectors.toarray(),columns=cv.get_feature_names_out())

            cs = cosine_similarity(vectors)
            ans = cs[0,1]
            ans1 = round(ans*100,2)

            msg = "Similarity is  :"+str(round(ans*100,2))+"%"
            if ans1 > 60 :
                msg = "Similarity is  :"+ str(ans1) +"%"
                background_color = "red"
            else:
                msg = "Similarity is  :"+ str(ans1) +"%"
                background_color = "green"
            styled_msg = f"<div style='text-align: center; background-color: {background_color}; padding: 10px;'>{msg}</div>"
            st.write(styled_msg, unsafe_allow_html=True)


            
