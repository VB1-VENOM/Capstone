# import nltk
# nltk.download('punkt')
# from nltk.tokenize import word_tokenize
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
#from models import InferSent
# import torch
# sentences = ["I ate dinner.", 
#        "We had a three-course meal.", 
#        "Brad came to dinner with us.",
#        "He loves fish tacos.",
#        "In the end, we all felt like we ate too much.",
#        "We all agreed; it was a magnificent evening."]
# tokenized_sent = []
# for s in sentences:
#     tokenized_sent.append(word_tokenize(s.lower()))
# print(tokenized_sent)

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

# def doc2vec(text):
#     tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(tokenized_sent)]
#     model = Doc2Vec(tagged_data, vector_size = 20, window = 2, min_count = 1, epochs = 100)
    
#     '''
#     vector_size = Dimensionality of the feature vectors.
#     window = The maximum distance between the current and predicted word within a sentence.
#     min_count = Ignores all words with total frequency lower than this.
#     alpha = The initial learning rate.
#     '''
#     print(model.wv)
#     test_doc = word_tokenize(text.lower())
#     test_doc_vector = model.infer_vector(test_doc)
#     # print("doc2vec : " , test_doc_vector)
#     return test_doc_vector

def sentenceBERT(text):
    
    sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
    query_vec = sbert_model.encode([text])[0]
    
    # print("sentenceBERT :  ", query_vec)
    return query_vec

# def inferSent(text):
#     '''Do these first:
#     ! mkdir encoder
#     ! curl -Lo encoder/infersent2.pkl https://dl.fbaipublicfiles.com/infersent/infersent2.pkl
    
#     ! mkdir GloVe
#     ! curl -Lo GloVe/glove.840B.300d.zip http://nlp.stanford.edu/data/glove.840B.300d.zip
#     ! unzip GloVe/glove.840B.300d.zip -d GloVe/
#     '''
#     V = 2
#     MODEL_PATH = 'encoder/infersent%s.pkl' % V
#     params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
#                     'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
#     model = InferSent(params_model)
#     model.load_state_dict(torch.load(MODEL_PATH))

#     W2V_PATH = './GloVe/glove.840B.300d.txt'
#     model.set_w2v_path(W2V_PATH)
#     model.build_vocab(sentences, tokenize=True)
#     query_vec = model.encode([text])[0]   
    
#     print("inferSent: ", query_vec)
#     return query_vec

def use(text):
    try:
        embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4") 

        def uni_encoder(texts):
            
            if type(texts) is str:
                texts = [texts]
            with tf.Session() as sess:
                sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
                return sess.run(embed(texts))

        #Universal encoder
        u1 = uni_encoder(text)[0]

        print("Universal sentence encoder : ", u1)
        return u1


    except:
        model_path = '/Users/ujwal_nischal/Desktop/Captsone/universal-sentence-encoder_4'  # Replace with the actual path to the directory

        # Load the SavedModel
        model = tf.saved_model.load(model_path)

        # Encode the text
        embeddings = model([text])

        # The 'embeddings' variable now contains the embedding for the input text
        # print("Google Sentence Encoder")
        # print("Embedding:", embeddings.numpy()[0])
        return embeddings.numpy()[0]
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
    return output[0]




# def test(text):
#     doc2vec(text)
#     sentenceBERT(text)
#     #inferSent(text)
#     use(text)
# n=input("Enter the sentence: ")
# test(n)