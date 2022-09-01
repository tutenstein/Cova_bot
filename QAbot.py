import json
import pandas as pd
import numpy as np
from Covabot import pre_process,cova_reply_bot
import gensim



stackoverflow_path = 'data/Word2Vec_data.json'

with open(stackoverflow_path) as file:
    reader = json.load(file)

    classes = []
    questions = []
    questions_tokens = []
    answers = []
    for row in reader:
        classes.append(row['Class'])
        questions.append(row['question'])
        questions_tokens.append(row['token'])
        answers.append(row['answer'])

    data_new = pd.DataFrame({'Class': classes,
                                    'question': questions,
                                    'token': questions_tokens,
                                    'answer': answers,})



GREETING_INPUTS = ("hello", "hi", "greetings", "hello i need help", "good day","hey","i need help", "greetings")
GREETING_RESPONSES = ["Good day, How may i of help?", "Hello, How can i help?", "hello", "I am glad! You are talking to me."]
           
def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)




flag_language = True
flag_query = True
dict_category = {'0': 'internship', '1': 'erasmus', '2': 'summer', '3': 'contact', '4': 'thesis', '5': 'other'}

print('......................................................................................')
print('\x1b[1;40;42m' + 'COVA' + '\x1b[0m' + ': ' + 'My name is COVA, a Çukurova University Computer Eng. Q&A Bot.')
print('\x1b[1;40;42m' + 'COVA' + '\x1b[0m' + ': ' + 'I will try my best to answer your query.')
print('\x1b[1;40;42m' + 'COVA' + '\x1b[0m' + ': ' + 'If you want to exit, you can type < bye >.')

while(flag_language == True):
    print("......................................................................................")
    print('\x1b[1;40;42m' + 'COVA' + '\x1b[0m' + ': ' + 'Please select which category you want to enquire, ' +
      'you can type:')
    print('\x1b[1;40;42m' + 'COVA' + '\x1b[0m' + ': ' + '< 0 > for internship    < 1 > for erasmus     < 2 > for summer')
    print('\x1b[1;40;42m' + 'COVA' + '\x1b[0m' + ': ' + '< 3 > for contact       < 4 > for thesis      < 5 > for other')
    print("......................................................................................")
    sentence = input('\x1b[0;30;44m' + 'USER  ' + '\x1b[0m' + ':')
    print("......................................................................................")
    
    if(sentence.lower() != 'bye'):
        if (sentence.lower() in list(dict_category.keys())):
            language = dict_category[sentence.lower()]
            data_language = data_new[data_new['Class'] == language]
            data_language = pd.DataFrame({'question': list(data_language['question']),
                                          'token': list(data_language['token']),
                                          'answer': list(data_language['answer']),
                                          'Class': list(data_language['Class']),
                                         })
            
            # Read word2vec model
            word2vec_pickle_path ='data/word2vec_' + language + '.bin'
            model = gensim.models.KeyedVectors.load(word2vec_pickle_path)
            
            flag_language = False
            flag_query = True
    else:
        flag_language = False
        flag_query = False

print("......................................................................................")
print('\x1b[1;40;42m' + 'COVA' + '\x1b[0m' + ': ' + 'Let''s start! Please input your question now.')
    
while(flag_query == True):
    print("......................................................................................")
    sentence = input('\x1b[0;30;44m' + 'USER  ' + '\x1b[0m' + ':')
    print("......................................................................................")

    if(sentence.lower() != 'bye'):
        if(greeting(sentence.lower()) != None):
            print('\x1b[1;40;42m' + 'COVA' + '\x1b[0m' + ': ' + greeting(sentence.lower()))
        else:
            reply, score,replies = cova_reply_bot(data_language, model,sentence)
            if(len(replies)!=0):
                print('\x1b[1;40;42m' + 'COVA'+'\x1b[0m'+': '+reply)
                print('\x1b[1;40;42m' + 'COVA'+'\x1b[0m'+': '+'Did you want to ask this questions?\n' +data_language['question'][replies['index'][1]]+'\n '+data_language['question'][replies['index'][2]])
            else:
                print('\x1b[1;40;42m' + 'COVA'+'\x1b[0m'+': '+reply)
            #For Tracing, comment to remove from print 
            #print("")
            #print("SCORE: " + str(score))
    else:
        flag_query = False
print('\x1b[1;30;42m' + 'COVA' + '\x1b[0m' + ': ' + 'Bye! Hope that i am of help.') 
