import streamlit as st
from sentiment_cardiff import get_sentiment
from summarization_bart import get_summary

# https://drive.google.com/uc?id=1qEFRPuwAJxxjORe-6vaC-WyLgTot_vaF
st.title("Welcome to NLP WORLD")

tab1, tab2 = st.tabs(["SENTIMENT ANALYSIS", "TEXT SUMMARIZATION"])

with tab1:
    st.header("TEXT INPUT FOR SENTIMENT ANALYSIS")
    if prompt := st.text_input(label=" ",placeholder=' Enter your text here to get the sentiment ...', key="a"):
            response = get_sentiment(prompt)
            st.header("Sentiment Analysis Result:")
            st.header(str(response).upper())
            st.write(" ")
            st.write("Model used for summarization: " + "RoBERTa")

with tab2:
    st.header("TEXT INPUT FOR SUMMARIZATION")
    if prompt := st.text_input(label=" ",placeholder=' Enter your text here to get the summary ...', key = "b"):
            response = get_summary(prompt)
            st.header("Summary Result: ")
            st.write(response)
            st.write(" ")
            st.write("Model used for summarization: " + "BART")





