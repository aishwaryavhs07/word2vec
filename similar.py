import os
import pickle
import numpy as np
from scipy import spatial


model_path = './models/'
#loss_model = 'cross_entropy'
loss_model = 'nce'

model_filepath = os.path.join(model_path, 'word2vec_%s.model'%(loss_model))

dictionary, steps, embeddings = pickle.load(open(model_filepath, 'rb'))

"""
==========================================================================

Write code to evaluate a relation between pairs of words.
You can access your trained model via dictionary and embeddings.
dictionary[word] will give you word_id
and embeddings[word_id] will return the embedding for that word.

word_id = dictionary[word]
v1 = embeddings[word_id]

or simply

v1 = embeddings[dictionary[word_id]]

==========================================================================
"""
res={}
list=[]
wordone=["first","american","would"]
for index in range(len(wordone)):
    chone= embeddings[dictionary[wordone[index]]]
    count=0
    for key in dictionary.keys():
        chtwo=embeddings[dictionary[key]]
        cosine_sim= 1- spatial.distance.cosine(chone,chtwo)
        res[cosine_sim]=key
        list.append(cosine_sim)
        
    print("word-",wordone[index])
    for key in sorted(res.keys(), reverse= True) :
        if count<=20:
            print(res[key],",")
            count=count+1
        
        else:
            break
    