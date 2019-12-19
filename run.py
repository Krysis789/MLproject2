#!/usr/bin/env python
# coding: utf-8

# In[]:
#IMPORT USED PACKAGES AND SET A SEED
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import *
import pickle
import random
from datetime import datetime
import pandas as pd
import csv
import pbs
import os
import sys
import seaborn as sns
import itertools
import json
from multiprocessing import Pool
from spellchecker import SpellChecker
from IPython.display import display
#IMPORTS NEEDED FOR BUILDING A CLASSIFIER
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
random.seed(123)





# In[]:
# Preprocessing

def is_not_auto_picture_tweet(string):
    if " frame " in string :
        beg, end = string.split(" frame ",1)
        return not ((" picture" in beg) & (" frame " in end))            
    else:
        return True

def drop_picture(input_filename):
    with open("Datasets/"+input_filename, encoding="utf-8") as f :
        tweets = f.read().splitlines()
        df_tweets = pd.DataFrame(tweets,columns=['Tweets'])
        df_tweets = df_tweets[df_tweets["Tweets"].apply(is_not_auto_picture_tweet)]
    with open('Processed_data/no_pic_' + input_filename, "w+", encoding = "utf-8") as f:
        for tweet in df_tweets["Tweets"]:
            f.write("%s\n" % tweet)
    
for file in ['train_neg_full.txt']:
    drop_picture(file)
    
# remove pictures from test data
with open("Datasets/test_data.txt", encoding="utf-8") as f :
    test = f.read().splitlines()
    tweets = []
    ids = []
    for line in test:
        id, tweet = line.split(',',1)
        tweets.append(tweet)
        ids.append(id)
    df_test = pd.DataFrame(tweets,ids,columns=['Tweets'])
    
    list_index_pictures_dropped_test = df_test[~df_test["Tweets"].apply(is_not_auto_picture_tweet)].index
    df_test = df_test[df_test["Tweets"].apply(is_not_auto_picture_tweet)]

    
with open('Processed_data/no_pic_test_data.txt', "w+", encoding = "utf-8") as f:
    for tweet in df_test["Tweets"]:
        f.write("%s\n" % tweet)

with open("Processed_data/no_pic_test_data_with_index.txt", "w+", encoding = "utf-8") as f:
    for index, row in df_test.iterrows():
        f.write(index + "," + row["Tweets"] + "\n")

with open("Processed_data/list_index_pictures_dropped_test.pkl", "wb") as f:
    pickle.dump(list_index_pictures_dropped_test, f, pickle.HIGHEST_PROTOCOL)



# In[]:
##### Remove duplicated tweets

def drop_duplicate(input_filename):
    with open(input_filename, encoding="utf-8") as f :
        tweets = f.read().splitlines()
        df_tweets = pd.DataFrame(tweets,columns=['Tweets'])
        df_tweets.drop_duplicates(inplace=True)
    with open('Processed_data/no_dupl_' + input_filename.split("/")[1], "w+", encoding = "UTF-8") as f:
        for tweet in df_tweets["Tweets"]:
            f.write("%s\n" % tweet)

for file in ['Processed_data/no_pic_train_neg_full.txt','Datasets/train_pos_full.txt']:
    drop_duplicate(file)


# drop duplicates test
with open("Processed_data/no_pic_test_data_with_index.txt", encoding="utf-8") as f :
    test = f.read().splitlines()
    tweets = []
    ids = []
    for line in test:
        id, tweet = line.split(',',1)
        tweets.append(tweet)
        ids.append(id)
    df_test = pd.DataFrame(tweets,ids,columns=['Tweets'])
    df_test.drop_duplicates(inplace=True)
    
with open('Processed_data/no_dupl_no_pic_test_data.txt', "w+", encoding = "UTF-8") as f:
    for tweet in df_test["Tweets"]:
        f.write("%s\n" % tweet)



# In[]:
##### Concatenate all thw text files (pos, neg and test) to create a dataset used as reference for twitter language

with open("Processed_data/twitter_language_full.txt", "w+", encoding = "utf-8") as f:
    for path in ['Processed_data/no_dupl_train_pos_full.txt','Processed_data/no_dupl_no_pic_train_neg_full.txt', 'Processed_data/no_dupl_no_pic_test_data.txt']:
        with open(path,encoding="utf-8") as file:
            f.write(file.read())


# In[]:

##Spell correction
## Create a frequency dictionnary of words occuring at least *threshold* times in the dataset, 
## that will be used as a reference for spell checking

output_filename = 'Processed_data/vocab_freq_full.txt'
vocab_successful = os.system("build_vocab.sh Processed_data/twitter_language_full.txt " + output_filename)

if (vocab_successful != 0):
    sys.exit("Building vocabulary failed.")


# In[]:

##Build a dictionary
correction_threshold = 20

def build_dictionary(input_vocab, threshold):
    full_vocab = pd.read_csv(input_vocab+".txt", sep = "\s+", header=None, engine='python')
    cut_vocab = full_vocab[full_vocab[0] >= threshold]
    cut_vocab.columns = ["number occ","word"]
    cut_vocab.set_index("word",inplace=True)
    with open(input_vocab + "_" + str(threshold) + '.json', 'w') as f:
        json.dump(cut_vocab["number occ"].to_dict(), f)

build_dictionary("Processed_data/vocab_freq_full", threshold = correction_threshold) 


# In[]:


## Spell correction functions
def correct(tweet, dict_corr):
    list_words = tweet.split()
    for i, word in enumerate(list_words):
        if word in dict_corr :
            list_words[i] = dict_corr[word]
    corr_tweet = ' '.join(list_words)
    return (corr_tweet) 

def spell_correction(data, reference_dictionnary_filename):
    spell = SpellChecker(distance=1) 
    spell.word_frequency.load_dictionary(reference_dictionnary_filename) 
    dict_corr = {}
    for tweet in data["Tweets"]:
        list_words = tweet.split()
        for i, word in enumerate(list_words):
            if word not in dict_corr :
                if word in spell.unknown([word]):
                    dict_corr[word] = spell.correction(word)
    data["Tweets"] = data["Tweets"].apply(lambda x : correct(x, dict_corr))
    return data, dict_corr


# In[]:


#Define the used datasets here
positive_dataset = 'Processed_data/no_dupl_train_pos_full.txt' #'Processed_data/no_dupl_train_pos_full.txt'
negative_dataset = 'Processed_data/no_dupl_no_pic_train_neg_full.txt' #'Processed_data/no_dupl_train_neg_full.txt'

def load_in_pd(data_path):
    with open(data_path, encoding="utf-8") as f :
        x = f.read().splitlines()
        x = pd.DataFrame(x,columns=['Tweets'])
    return (x)

positive_pd = load_in_pd(positive_dataset)
negative_pd = load_in_pd(negative_dataset)


# In[]:

### APPLYING spell correction to pos and neg

positive_preprocessed, dict_corr1 = spell_correction(positive_pd, "Processed_data/vocab_freq_full" + "_" + str(correction_threshold) + '.json') #'Processed_data/vocab_freq_full.json') 
with open('Processed_data/' + 'positive_spell_full.txt', "w+", encoding = "UTF-8") as f: #positive_spell_full.txt', "w+", encoding = "UTF-8") as f:
    for tweet in positive_preprocessed["Tweets"]:
        f.write("%s\n" % tweet)
del positive_preprocessed
print("done positive")

negative_preprocessed, dict_corr2 = spell_correction(negative_pd, "Processed_data/vocab_freq_full" + "_" + str(correction_threshold) + '.json') #'Processed_data/vocab_freq_full.json') 
with open('Processed_data/' + 'negative_spell_full.txt', "w+", encoding = "UTF-8") as f: #negative_spell_full.txt', "w+", encoding = "UTF-8") as f:
    for tweet in negative_preprocessed["Tweets"]:
        f.write("%s\n" % tweet)
del negative_preprocessed


# In[]:


### APPLYING spell correction to test

test_dataset = "Datasets/test_data.txt"

def load_in_pd_test(data_path):
    with open(data_path, encoding="utf-8") as f :
        test = f.read().splitlines()
        tweets = []
        ids = []
        for line in test:
            id, tweet = line.split(',',1)
            tweets.append(tweet)
            ids.append(id)
        test = pd.DataFrame(tweets,ids,columns=['Tweets'])
    return test

test_pd = load_in_pd_test(test_dataset)
test_spell, dict_corr_test = spell_correction(test_pd,"Processed_data/vocab_freq_full" + "_" + str(correction_threshold) + '.json') #'Processed_data/vocab_freq_full.json')  

with open("Processed_data/test_spell.txt", "w+", encoding="utf-8") as f:
    for index, row in test_spell.iterrows():
        f.write(index + "," + row["Tweets"] + "\n")


# In[]:


# gather input for word embeddings
with open("Processed_data/input_WE_full.txt", "w+", encoding = "utf-8") as f:
    with open("Processed_data/test_spell.txt",encoding="utf-8") as test:
        test = test.readlines()
        for line in test:
            id, tweet = line.split(',',1)
            f.write("%s" % tweet)
    
    for path in ['Processed_data/positive_spell_full.txt','Processed_data/negative_spell_full.txt']:
        with open(path,encoding="utf-8") as file:
            f.write(file.read())


# In[]:

##APPLY build_vocab.sh AND CUT THE VOCABULARY USING THE CHOSEN THRESHOLD

output_filename = 'Processed_data/vocab_freq_after_spell_full.txt' #'Processed_data/vocab_freq_after_spell_full.txt'
vocab_successful = os.system("build_vocab.sh Processed_data/input_WE_full.txt " + output_filename)

if (vocab_successful != 0):
    sys.exit("Building vocabulary failed.")


cut_threshold = 20

output_filename = 'Processed_data/vocab_full'  + '.txt'
vocab_successful = os.system("build_vocab.sh Processed_data/input_WE_full.txt " + output_filename)

if (vocab_successful != 0):
    sys.exit("Building vocabulary failed.")
    
def cut_and_save_vocab(file_in, file_out):
    full_vocab = pd.read_csv(file_in, sep = "\s+", header=None, engine = 'python')
    cutted_vocab = full_vocab[full_vocab[0] >= cut_threshold][1]
    with open(file_out, 'w+') as f:
        f.write(cutted_vocab.to_string(header = False, index = False))
    
cut_and_save_vocab('Processed_data/vocab_full' + '.txt', 'Processed_data/vocab_cut_full' + '.txt')
    

##DUMP THE BUILT VOCABULARY TO A PICKLE FILE
vocab = dict()
with open('Processed_data/vocab_cut_full' + '.txt') as f:
    for idx, line in enumerate(f):
        vocab[line.strip()] = idx

with open('Processed_data/vocab_full' + '.pkl', 'wb') as f:
    pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)



##CREATE A CO-OCCURRENCE MATRIX with context defined with threshold distance
def context(lst,index,threshold):
    return(lst[index+1 : min(index+threshold, len(lst)-1)])

def create_cooc_dist(vocab_file, negative_file, positive_file, output_file, threshold):
    with open(vocab_file, 'rb') as f:
        vocab = pickle.load(f)
    data, row, col = [], [], []
    counter = 1
    for fn in [negative_file, positive_file]:
        with open(fn,encoding="utf8") as f:
            for line in f:
                tokens = [vocab.get(t, -1) for t in line.strip().split()]
                #tokens = [t for t in tokens if t >= 0]
                for i, t1 in enumerate(tokens):
                    if t1 >= 0:
                        for t2 in context(tokens,i,threshold):
                            if t2 >= 0:
                                if t1==t2 :
                                    data.append(1/2)
                                else :
                                    data.append(1)
                                row.append(min(t1,t2))
                                col.append(max(t1,t2))

                if counter % 100000 == 0:
                    print(counter)
                counter += 1

                if counter % 3000000 == 0:
                    print(len(data))
                    cooc = coo_matrix((data, (row, col)))
                    print("Summing duplicates")
                    cooc.sum_duplicates()
                    data=list(cooc.data)
                    row=list(cooc.row)
                    col=list(cooc.col)
                    print(len(data))

    print(len(data))
    final_row = row+col
    final_col = col+row
    data = data+data
    cooc = coo_matrix((data, (final_row, final_col)))
    print("Summing duplicates")
    cooc.sum_duplicates()
    
    with open(output_file, 'wb') as f:
        pickle.dump(cooc, f, pickle.HIGHEST_PROTOCOL)
        
#Used datasets should be defined on the second code block
context_threshold = 2    #for context_threshold in [2,5,144] :
create_cooc_dist('Processed_data/vocab_full' + '.pkl', 'Processed_data/negative_spell_full.txt', 'Processed_data/positive_spell_full.txt',
            'Processed_data/cooc_dist_pickle_full_threshold_' + str(context_threshold) + '.pkl', context_threshold)

# In[]:


#Do not create word embeddings for these stop words
def drop_stop_rows(cooc_path, stop_path, vocab_path, output_path):
    with open(cooc_path, 'rb') as f:
        cooc = pickle.load(f)
    with open(stop_path, encoding = "utf-8") as stoplist:
        words = stoplist.read().splitlines()
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    stop_indexes = [vocab.get(t, -1) for t in words]
    keeps = np.logical_not(np.isin(cooc.row, stop_indexes))
    new_cols = cooc.col[keeps]
    new_rows = cooc.row[keeps]
    new_data = cooc.data[keeps]
    new_cooc = coo_matrix((new_data, (new_rows, new_cols)))
    with open(output_path, 'wb') as f:
        pickle.dump(new_cooc, f, pickle.HIGHEST_PROTOCOL)

drop_stop_rows('Processed_data/cooc_dist_pickle_full_threshold_' + str(context_threshold) + '.pkl', "Stoplist.txt", 
              'Processed_data/vocab_full' + '.pkl', 'Processed_data/cooc_dist_pickle_cutted_threshold_' + str(context_threshold) + '.pkl')      


# In[]:


## CREATE WORD EMBEDDINGS
random.seed(123)
def glove(cooc_pickle, output_file, embedding_dim):
    print("loading cooccurrence matrix")
    with open(cooc_pickle, 'rb') as f:
        cooc = pickle.load(f)
    print("{} nonzero entries".format(cooc.nnz))

    nmax = 100
    print("using nmax =", nmax, ", cooc.max() =", cooc.max())

    print("initializing embeddings")
    xs = np.random.normal(size=(cooc.shape[0], embedding_dim))
    ys = np.random.normal(size=(cooc.shape[1], embedding_dim))

    eta = 0.001
    alpha = 3 / 4

    epochs = 15

    for epoch in range(epochs):
        print("epoch {}".format(epoch))
        for ix, jy, n in zip(cooc.row, cooc.col, cooc.data):
            logn = np.log(n)
            fn = min(1.0, (n / nmax) ** alpha)
            x, y = xs[ix, :], ys[jy, :]
            scale = 2 * eta * fn * (logn - np.dot(x, y))
            xs[ix, :] += scale * y
            ys[jy, :] += scale * x
    np.save(output_file, xs)

embedding_dim = 100
glove('Processed_data/cooc_dist_pickle_cutted_threshold_' + str(context_threshold) + '.pkl',
      'Processed_data/embeddings_dim_' + str(embedding_dim) + "_context_" + str(context_threshold) + '.npy', embedding_dim)


# In[]:


##### LOAD THE TEST DATA, USE THE MODEL TO PREDICT THE SENTIMENTS AND SAVE THE PREDICTIONS
with open('Processed_data/vocab_full' + '.pkl', 'rb') as f:
    vocab = pickle.load(f)
f.close()
we = np.load('Processed_data/embeddings_dim_' + str(embedding_dim) + "_context_" + str(context_threshold) + '.npy')

#Define the used datasets here
positive_path = 'Processed_data/positive_spell_full.txt'
negative_path = 'Processed_data/negative_spell_full.txt'

def load_train_data(data_path):
    """Loads data and returns ids (event ids) and X (feature vector)"""
    f = open(data_path, encoding="utf-8")
    x = f.readlines()
    x = pd.DataFrame(x, columns=['Tweets'])
    f.close()
    return x

def load_and_label(negative_path, positive_path):
    pos = load_train_data(positive_path)
    neg = load_train_data(negative_path)
    pos["y"] = 1
    neg["y"] = 0
    train = pd.concat([pos, neg])
    train.reset_index(drop = True, inplace = True)
    return train


train = load_and_label(negative_path, positive_path)

def count_tokens(line, vocab):
    tokens = line.split()
    nof_tokens = 0
    for token in tokens:
        if token in vocab.keys():
            nof_tokens += 1
    return nof_tokens

train["nof_tokens"] = train["Tweets"].apply(lambda x: count_tokens(x, vocab))

max_tokens = max(train["nof_tokens"])
max_tokens = 25
train = train[train["nof_tokens"] <= max_tokens]
train.reset_index(drop = True, inplace = True)

def data_to_matrix_array(train, max_tokens, we, vocab, stop_path):
    with open(stop_path, encoding = "utf-8") as stoplist:
        stop_words = stoplist.read().splitlines()
    train_data = np.zeros((len(train), max_tokens, np.shape(we)[1]), dtype = np.float32)
    i = -1
    stop_indexes = [vocab.get(t, -1) for t in stop_words]
    valid_keys = list(filter(lambda x: x not in stop_indexes, vocab.keys()))
    filtered_vocab = {key: vocab[key] for key in valid_keys}
    for line in train["Tweets"]:
        i += 1
        if i % 20000 == 0:
            print(i)
        count = max_tokens
        tokens = line.split()
        for t in tokens:
            if t in filtered_vocab.keys():
                count -= 1
                train_data[i, count, :] = we[filtered_vocab[t]]
    return train_data
    

train_data = data_to_matrix_array(train, max_tokens, we, vocab, 'Stoplist.txt')
            


# In[]:

##Define the classifier that will be fit

def create_model(max_tokens, dim):
    model = Sequential()
    model.add(LSTM(60, input_shape = (max_tokens, dim), return_sequences = True))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(10, activation = 'sigmoid'))
    model.add(Dense(1, activation = 'sigmoid'))

    # Compile model
    model.compile(
        loss = 'mse',
        optimizer = 'adam',
        metrics = ['accuracy'],
    )
    return model


# In[]:


tensorflow.random.set_seed(112233)
np.random.seed(123)
classifier = create_model(max_tokens, np.shape(we)[1])
classifier.fit(x = train_data, y = np.asarray(train["y"]), batch_size=4, epochs=2, verbose=1, shuffle = True)


# In[]:

##### LOAD THE TEST DATA, USE THE MODEL TO PREDICT THE SENTIMENTS AND SAVE THE PREDICTIONS

def load_test_data(data_path):
    """Loads data and returns ids (event ids) and X (feature vector)"""
    f = open(data_path, encoding="utf-8")
    x = f.readlines()
    x = pd.DataFrame(x, columns=['Tweets'])
    f.close()
    return x


test = load_test_data("Processed_data/test_spell.txt")

test_data = data_to_matrix_array(test, max_tokens, we, vocab, 'Stoplist.txt')


# In[]:


# CREATE THE SUBMISSION FILE
predictions = np.reshape(classifier.predict(test_data), -1)
predictions[predictions > 0.5] = 1
predictions[predictions < 1] = -1
test["Id"] = range(1, 1 + len(predictions))
test["Prediction"] = np.round(predictions)

# automatic picture tweets have to be classified as negative
with open("Processed_data/list_index_pictures_dropped_test.pkl", 'rb') as f:
    list_index_pictures_dropped_test = pickle.load(f)
for index in list_index_pictures_dropped_test :
    test.at[int(index),"Prediction"] = -1

test[["Id", "Prediction"]].to_csv("Submissions/submission_dim" + str(embedding_dim) + "_cut" + str(cut_threshold) + "_corr" +
                                  str(correction_threshold) + "_context" + str(context_threshold)+ ".csv", header= True, index = False, float_format='%.0f')



