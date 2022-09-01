import json
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
import os
import json
import re
import numpy as np
import nltk
from nltk.corpus import stopwords 
from nltk import pos_tag
import gensim
from gensim import corpora, models, similarities
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
from nltk import ngrams
from collections import Counter

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns



def pre_process(questions):
    stop_words = stopwords.words("english")
    stop_words.append("n't")
    stop_words.append('ca')
    stop_words.append("n't")
    stop_words.append('s')
    
    questions = [re.sub('[^a-z]', ' ', x.lower()) for x in questions]
    questions_tokens = [nltk.word_tokenize(t) for t in questions]
    questions_stop = [[t for t in tokens if (t not in stop_words)]
                      for tokens in questions_tokens]
            
    questions_stops = []
    for i in questions_stop:
        lem_qu = []
        for a in i:
            lem_qu += [WordNetLemmatizer().lemmatize(a)]
        questions_stops += [lem_qu]
    questions_stop = pd.Series(questions_stops)
    
    return questions_stops

def word_frequency(questionpp):
    new_tokens = []
    for i in questionpp:
        for a in i:
            new_tokens+=[a]
    counted = Counter(new_tokens)
    counted_2= Counter(ngrams(new_tokens,2))
    counted_3= Counter(ngrams(new_tokens,3))
    word_freq = pd.DataFrame(counted.items(),columns=['word','frequency']).sort_values(by='frequency',ascending=False)
    word_pairs =pd.DataFrame(counted_2.items(),columns=['pairs','frequency']).sort_values(by='frequency',ascending=False)
    trigrams =pd.DataFrame(counted_3.items(),columns=['trigrams','frequency']).sort_values(by='frequency',ascending=False)
    return word_freq,word_pairs,trigrams




categories = ['internship','erasmus','summer school','Contact information','graduation thesis','others']

def create_category(data,key):
    question = []
    answer = []
    token = []
    for i in range(len(data)):
        if(key in data['token'][i]):
            question += [data['question'][i]]
            answer += [data['answer'][i]]
            token += [data['token'][i]]
    df_dict = {'question': question, 'answer': answer, 'token': token}
    df = pd.DataFrame(df_dict)
    df['Class'] = key
    return df
def create_other_category(data,keys):
    question = []
    answer = []
    token = []
    for i in range(len(data)):
        if(keys[0]in data['token'][i]):
            continue
        elif(keys[1]in data['token'][i]):
            continue
        elif(keys[2]in data['token'][i]):
            continue
        elif(keys[3]in data['token'][i]):
            continue
        elif(keys[4]in data['token'][i]):
            continue
        else: 
            question += [data['question'][i]]
            answer += [data['answer'][i]]
            token += [data['token'][i]]
    df_dict = {'question': question, 'answer': answer, 'token': token}
    df = pd.DataFrame(df_dict)
    df['Class'] = 'other'
    return df



def train_model(train_data):
    model = gensim.models.Word2Vec(train_data, min_count=1)
    return model


dict_category = {'0': 'internship', '1': 'erasmus', '2': 'summer', '3': 'contact', '4': 'thesis', '5': 'other'}



def cova_reply_bot(data_language, model,sentence):
    
    sentence_pp = pre_process(pd.Series(sentence)) 
    first = 0
    score = []
    index = []

    for i in range(len(data_language['token'])):
        tokens= data_language['token'][i]
        a = model.wv.n_similarity(tokens, sentence_pp[0])
        if (a>0.5):
            score += [a]
            index += [i]
            
    
    if len(index)==0:
        not_understood = "Apology, I do not understand. Can you rephrase?"
        return not_understood, 999,[]
    
    else: 
            
        replies={ 'score':score,
             'index': index}
        replies = pd.DataFrame(replies).sort_values(by=['score'],ascending=False,ignore_index=True)
        r_index = int(replies['index'][0])
        r_score = float(replies['score'][0])
        reply = str(data_language['answer'][r_index])
        
        return reply, r_score,replies

