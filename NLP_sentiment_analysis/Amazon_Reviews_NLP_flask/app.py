#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 17:46:20 2019

@author: srinivasgurrala
"""

from flask import Flask,render_template,url_for,request
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer,WordNetLemmatizer
from nltk import word_tokenize
import re
import pandas as pd
import contractions
import inflect
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from tabulate import tabulate

stop_words = stopwords.words('english') # remove stop words

def get_percentage(num):
    return "{:.2f}%".format(num*100)

def number_to_text(data):
    temp_str = data.split()
    string = []
    for i in temp_str:

    # if the word is digit, converted to
    # word else the sequence continues

        if i.isdigit():
            temp = inflect.engine().number_to_words(i)
            string.append(temp)
        else:
            string.append(i)
    outputStr = " ".join(string)
    return outputStr

ps = PorterStemmer()
def stem_text(data):
    tokens = word_tokenize(data)
    stemmed_tokens = [ps.stem(word) for word in tokens if word not in (stop_words)]
    return " ".join(stemmed_tokens)

lemma = WordNetLemmatizer()
def lemmatiz_text(data):    
    tokens = word_tokenize(data)
    lemma_tokens = [lemma.lemmatize(word, pos='v') for word in tokens if word not in (stop_words)]
    return " ".join(lemma_tokens)

def cleantext(text):
    text = re.sub(r'[^\w\s]', " ", text) # Remove punctuations
    text = re.sub(r"https?:\/\/\S+", ",", text) # Remove The Hyper Lin
    text = contractions.fix(text) # remove contractions 
    text = number_to_text(text) # convert numbers to text    
    text = text.lower() # convert to lower case
    # don't feel it's worth to use stemming as it may lead to some wrong words
    text = lemmatiz_text(text) # lemmatization
    return text
    

app = Flask(__name__)

#decorator (@app.route('/')) to specify the URL that should trigger the execution of the home function

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	
    pickle_in = open("model_sentiment.pkl", 'rb') 
    model_sentiment = pickle.load(pickle_in)

    pickle_in_tdidf = open("model_sentiment_tfidf.pkl", 'rb') 
    model_tfidf = pickle.load(pickle_in_tdidf)
    
    amazonreviewText = request.form['review']
    
    cleanReviewText = cleantext(amazonreviewText)
    #st.write(cleanReviewText)    
    tfIdfText = model_tfidf.transform([cleanReviewText])
    predictedVal=model_sentiment.predict(tfIdfText)
    predictedVal= predictedVal[0]

    dfRes = pd.DataFrame(columns=['Negative', 'Positive'])
    
         
    prdictionDist = model_sentiment._predict_proba_lr(tfIdfText)

    dfRes.loc[1, 'Negative'] = get_percentage(prdictionDist[0][0])
    dfRes.loc[1, 'Positive'] = get_percentage(prdictionDist[0][1])
        

    return render_template('result.html',predictedVal = predictedVal, dataProb=dfRes.to_html(index = False),dataProbOrg=dfRes)

if __name__ == '__main__':
	app.run(debug=True)