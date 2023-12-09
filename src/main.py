import streamlit as st
from sentiment_cardiff import get_sentiment
from summarization_bart import get_summary

# https://drive.google.com/uc?id=1qEFRPuwAJxxjORe-6vaC-WyLgTot_vaF
st.title("Welcome to NLP WORLD")

tab1, tab2, tab3 = st.tabs(["HOME PAGE","SENTIMENT ANALYSIS", "TEXT SUMMARIZATION"])

with tab1:
    st.header("NLP RELATED TASK DEVELOPMENT")

with tab2:
    if prompt := st.text_input(label="TEXT INPUT FOR SENTIMENT ANALYSIS",placeholder=' Enter your text here to get the sentiment ...', key="a"):
            response = get_sentiment(prompt)
            st.header("Sentiment Analysis Result:")
            st.header(str(response).upper())
            st.write(" ")
            # st.write("Model used for summarization: " + "RoBERTa")

with tab3:
    if prompt := st.text_input(label="TEXT INPUT FOR SUMMARIZATION",placeholder=' Enter your text here to get the summary ...', key = "b"):
            response = get_summary(prompt)
            st.header("Summary Result: ")
            st.write(response)
            st.write(" ")
            # st.write("Model used for summarization: " + "BART")





