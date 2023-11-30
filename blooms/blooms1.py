import csv
from llms import *
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sentence_transformers import SentenceTransformer
import numpy as np
import tensorflow as tf

import tensorflow_hub as hub
from sentence_transformers import SentenceTransformer
import nltk
import requests
import re
import pandas as pd
import plotly.express as px
csv_file_path = 'testdataset.csv'
df=pd.read_csv(csv_file_path)
def mini_llm(answer,llmanswer):
    API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
    headers = {"Authorization": "Bearer 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"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()
        
    output = query({
        "inputs": {
            "source_sentence": answer,
            "sentences": [
                llmanswer
            ]
        },
    })
    return output
#write function to get the average of the values in a column group by column called as 'Type' using dictReader object

def get_average(column_name,df):
    #get the unique values of column 'Type'
    unique_values = df['Type'].unique()
    #create an empty dictionary
    average_dict = {}
    #iterate through the unique values
    for value in unique_values:
        #get the average of the column 'Type' where the value of column 'Type' is equal to the unique value
        average = df[df['Type'] == value][column_name].mean()
        print(average)
        #add the average to the dictionary
        average_dict[value] = average
    #return the dictionary
    return average_dict
def graph(dict):
    df1=pd.DataFrame(dict.items(),columns=['Type','Average'])
    print(dict)
    print('--------------------')
    print(df1)
    fig = px.line_polar(df1, r='Average', theta='Type', line_close=True)
    fig.show()

with open(csv_file_path, 'r') as file:
    reader = csv.DictReader(file)

    for i in range(len(df)):
        question=df.at[i,'Question']
        df.at[i, 'ChatGPT'] = chatgpt(question)
        df.at[i, 'ChatGPTSimilarity'] = mini_llm(question,df.at[i, 'ChatGPT'])
        df.at[i, 'Llama'] = llama(question)
        df.at[i, 'LlamaSimilarity'] = mini_llm(question,df.at[i, 'Llama'])
        df.at[i, 'Mistral'] = mistral(question)
        df.at[i, 'MistralSimilarity'] = mini_llm(question,df.at[i, 'Mistral'])
        df.at[i, 'Zephyr'] = obs(question)
        df.at[i, 'ZephyrSimilarity'] = mini_llm(question,df.at[i, 'Zephyr'])
        print(df.at[i, 'ChatGPT'])
        print(df.at[i, 'ChatGPTSimilarity'])
        print(df.at[i, 'Llama'])
        print(df.at[i, 'LlamaSimilarity'])
        print(df.at[i, 'Mistral'])
        print(df.at[i, 'MistralSimilarity'])
        print(df.at[i, 'Zephyr'])
        print(df.at[i, 'ZephyrSimilarity'])
        print('---------------------------')
    #print(reader)
    # for row in df.index:
    #     question = df['Question'][row]
    #     print(type(question))
    #     # Query the data using chatGPT
    #     # BEGIN: code for chatGPT query
    #     df['ChatGPT'][row]=chatgpt(question)
    #     print(df['ChatGPT'][row])
    #     #print(row['ChatGPT'])
    #     df['ChatGPTSimilarity'][row]=mini_llm(question,df['ChatGPT'][row])
    #     print(df['ChatGPTSimilarity'][row])
    #     #print(row['ChatGPTSimilarity'])

    #     # END: code for chatGPT query
        
        # Query the data using Llama
        # BEGIN: code for Llama query
        # row['Llama']=llama(question)
        # print(row['Llama'])
        # row['LlamaSimilarity']=mini_llm(question,row['Llama'])
        # print(row['LlamaSimilarity'])
        
        # # END: code for Llama query
        
        # # Query the data using MistralAI
        # # BEGIN: code for MistralAI query
        # row['Mistral']=mistral(question)
        # print(row['Mistral'])
        # row['MistralSimilarity']=mini_llm(question,row['Mistral'])
        # print(row['MistralSimilarity'])
        
        # # END: code for MistralAI query
        
        # # Query the data using Zephyr
        # # BEGIN: code for Zephyr query
        # row['Zephyr']=obs(question)
        # print(row['Zephyr'])
        # row['ZephyrSimilarity']=mini_llm(question,row['Zephyr'])
        # print(row['ZephyrSimilarity'])
        
        # # END: code for Zephyr query

graph(get_average('ChatGPTSimilarity',df))
graph(get_average('LlamaSimilarity',df))
graph(get_average('MistralSimilarity',df))
graph(get_average('ZephyrSimilarity',df))
