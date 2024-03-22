import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))


def transform_text(text):
    text = text.lower()  # convert all letter in lower case
    text = nltk.word_tokenize(text)  # break the sentence (tokanize)

    y = []
    for i in text:
        if i.isalnum():  # remove all the special char
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        if i not in nltk.corpus.stopwords.words('english'):  # remove all the stop words
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in string.punctuation:  # remove all the punctuation
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        y.append(nltk.PorterStemmer().stem(i))  # stemming
    return " ".join(y)

st.title("Sms spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):

    #1.preprocess
    transform_sms = transform_text(input_sms)
    #2.vectorize
    vector_input = tfidf.transform([transform_sms])
    #3.predict
    result = model.predict(vector_input)[0]
    #4.display
    if result==1:
        st.header("The message is a spam")
    else:
        st.header("The message is not a spam")