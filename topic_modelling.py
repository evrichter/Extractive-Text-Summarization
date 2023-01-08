#!/usr/bin/env python
# coding: utf-8

from pprint import pprint
import os, re, sys
from nltk.corpus import stopwords
from string import punctuation
from string import digits
from gensim.models import Phrases
import pickle
import spacy
nlp = spacy.load("de_core_news_sm")

import pandas as pd

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
from gensim import models, corpora
import numpy as np
from documentSummaries import inference_summary
import datetime
import argparse

from rogue_r import get_rogue
a = datetime.datetime.now()


g_num_topics = 20
g_min_word_count = 10
g_top_most_common_words=25
g_min_doc_length=10
g_max_doc_length=1000
g_random_state=None
stop_words = stopwords.words('german')
bigramizer = Phrases()



def tokenizer(document):
    """
    """
    text = re.sub('[^A-Za-zÀ-ž\u0370-\u03FF\u0400-\u04FF]', ' ', document)
    tokens = text.lower().split()
    doc = nlp(" ".join(tokens))
    tokens = [x.lemma_ for x in doc]
    return tokens


def count_frequencies(tokens):
        """
        input: tokens, a list of list of tokens
        output: a collections.Counter() object that contains token counts
        """
        frequencies = Counter()
        for row in tokens:
            frequencies.update(row)
        return frequencies
    

def preprocessing( documents, min_word_count=None, 
                        top_most_common_words=None, min_doc_length=None, 
                        max_doc_length=None):
       
       min_word_count = g_min_word_count
       top_most_common_words = g_top_most_common_words
       min_doc_length = g_min_doc_length
       max_doc_length = g_max_doc_length
       
       tokens = [tokenizer(document) for document in documents]
       tokens = [tkn for tkn in tokens if len(tkn) < max_doc_length]
       bigramizer.add_vocab(tokens)
       tokens = [bigramizer[tkn] for tkn in tokens]
       tokens = [[t for t in tkn if t not in stop_words] for tkn in tokens]
       tokens = [tkn for tkn in tokens if len(tkn) > min_doc_length]
       freqs = count_frequencies(tokens)
       low_freq_tokens = set(x[0] for x in freqs.items() if x[1] < min_word_count)
       high_freq_tokens = [word[0] for word in freqs.most_common(top_most_common_words)]
       tokens =  [[t for t in tkn if t not in low_freq_tokens] for tkn in tokens]
       tokens =  [[t for t in tkn if t not in high_freq_tokens] for tkn in tokens]
       
       return tokens

def divide_chunks(l, n):

    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]

def divide_chunks(l, n):

    for i in range(0, len(l), n):
        yield l[i:i + n]


def topic_modelling(train_path):
    
    df = pd.read_csv(train_path)
    tokens = preprocessing(df["source"].tolist())
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(text) for text in tokens]
    lda = models.ldamodel.LdaModel(corpus=corpus, 
                                       alpha='auto', 
                                       id2word=dictionary, 
                                       num_topics=g_num_topics,
                                       random_state=g_random_state)
    return dictionary, lda


def save_model( dictionary,lda):

    lda_path = "output/lda_model"
    dict_path = "output/dictionary.p"
    lda.save(lda_path)

    with open(dict_path, 'wb') as fp:
        pickle.dump(dictionary, fp, protocol=pickle.HIGHEST_PROTOCOL)

    return lda_path, dict_path


def load_model(lda_path, dict_path):

    lda = models.LdaModel.load(lda_path)
    f = open(dict_path,'rb')
    dictionary = pickle.load(f)
    f.close()

    return lda, dictionary
    


def test_summary(lda,dictionary, test_path):
    
    df = pd.read_csv(test_path)
    hypothesis = []
    references = []
    for index, row in df.iterrows():
        chunk_size = 10
        sentences = row['source'].split(".")
        expected_summary = "\n".join(row["summary"].split("."))
        x = list(divide_chunks(sentences, chunk_size))
        docSummaries = inference_summary(lda, dictionary, num_dominant_topics=5, number_of_sentences=4)
        docSummaries.summarize([".".join(xx) for xx in x])
        predicted_summary = "\n".join([x.replace("'", "") for x in list(docSummaries.summary_data)])
        hypothesis.append(predicted_summary)
        references.append(expected_summary)
    get_rogue(hypothesis, references)



def get_summary(lda, dictionary, text):

    chunk_size = 10
    x = list(divide_chunks(text.split("."), chunk_size))

    docSummaries = inference_summary(lda, dictionary, num_dominant_topics=5, number_of_sentences=4)
    docSummaries.summarize([".".join(xx) for xx in x])
    predicted_summary = "\n".join([x.replace("'", "") for x in list(docSummaries.summary_data)])
    print("Predicted Summary is:")
    print(predicted_summary)



if __name__=="__main__":


    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("type", help="either train, test or both or inference")
    parser.add_argument("-train_file", default = "data/final_train.csv",help="path to train file")
    parser.add_argument("-test_seen", default="data/seen_test.csv",help="path to seen test file")
    parser.add_argument("-test_unseen",default="data/unseen_test.csv", \
                        type=str,help='path to unseen test file')
    parser.add_argument("-model_path",default="output/lda_model", \
                        type=str,help='path to saved model or path to save model')
    parser.add_argument("-dict_path",default="output/dictionary.p",choices=['log'], \
                        type=str,help='path to saved model or path to save dictionary')
    parser.add_argument("-document", help="documemt you wanna summarize in a text file")

    args = parser.parse_args()
    
    if args.type == "train":

        dictionary,lda = topic_modelling(args.train_file)
        print("model saved")
        dictionary_path = save_model( dictionary, lda)
        import sys
        sys.exit("done")
    elif args.type == "both":
        dictionary,lda = topic_modelling(args.train_file)
        print("model saved")
        dictionary_path = save_model( dictionary)
        lda, dictionary = load_model(lda_path, dict_path)
        print("model loaded")
        test_summary(lda, dictionary, args.test_unseen)
        test_summary(lda, dictionary, args.test_seen)

    elif args.type == "test":

        lda, dictionary = load_model(args.model_path, args.dict_path)
        print("model loaded")
        #pprint(lda.print_topics())
        print("TEST FOR UNSEEN DOCUMENT")
        test_summary(lda, dictionary, args.test_unseen)
        print("TEST FOR SEEN DOCUMENT")
        test_summary(lda, dictionary, args.test_seen)
    elif args.type == "inference":

        if not args.document:
            sys.exit("Please enter the text as well")

        else:
            lda, dictionary = load_model(args.model_path, args.dict_path)
            print("model loaded")
            text = open(args.document, "r").read()
            print("document to be summarized:")
            print(text)
            get_summary(lda, dictionary, text)

