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
input= open("word_analogy_test.txt",'r') #read the word_analogy_dev.txt file
output= open("word_analogy_test_predictions_nce.txt",'w') #write into an output text file

for line in input:  #gets each line in the input file
    cosine_sim_list=[]
    examples,choices = line.strip().split('||') # get examples and choices for each line
    #print("examples",examples)
    #print("choices",choices)
    examplesplit= examples.strip().split(",")
    #print(len(examplesplit))
    #print("example split:",examplesplit)
    avglist=[]
    for ex in range(len(examplesplit)):
        wordone,wordtwo= examplesplit[ex].strip('"').split(":") #get the two words in each example pair by removing " "
        #print("wordone:",wordone)
        #print("wordtwo:",wordtwo)
        #obtain word embeddings
        embeddingone= embeddings[dictionary[wordone]]
        embeddingtwo= embeddings[dictionary[wordtwo]]
        avglist.append(embeddingone-embeddingtwo) 
        
    avgembed = np.mean(avglist)
    
    #calculating for choices
    choicesplit= choices.strip().split(",")
    for ch in range(len(choicesplit)):
        firstword,secondword= choicesplit[ch].strip('"').split(":") #get the two words in each example pair by removing " "
        chone= embeddings[dictionary[firstword]]
        chtwo= embeddings[dictionary[secondword]]
        choicediff= chone-chtwo
        cosine_sim= 1- spatial.distance.cosine(avgembed,choicediff)
        cosine_sim_list.append(cosine_sim) #cosine similarity list
        output.write(choicesplit[ch]+" ")
        #print(choicesplit[ch])
    leastindex=np.argmin(cosine_sim_list) #get the least value in the list
    output.write(choicesplit[leastindex]+" ")
    highestindex=np.argmax(cosine_sim_list) #get the highest value in the list
    output.write(choicesplit[highestindex]+"\n")
        
output.close()
    