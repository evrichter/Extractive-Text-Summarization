import re
import nltk.data
import numpy as np
from collections import defaultdict
from textrankmaster.text_rank.text_rank import summarization


import spacy
nlp = spacy.load("de_core_news_sm")


def tokenizer(document):
    """
    input: a string
    output: a list of strings
    converts a string into tokens and performs the following steps:
    1. elimaintes non alphabetical characters
    2. converts to lower case
    3. lemmatizes using the nltk.stem.WordNetLemmatizer
    4. splits into tokens
    """
    text = re.sub('[^A-Za-zÀ-ž\u0370-\u03FF\u0400-\u04FF]', ' ', document)
    tokens = text.lower().split()
    doc = nlp(" ".join(tokens))
    tokens = [x.lemma_ for x in doc]
    return tokens


class topicSummary(object):

    def __init__(self, topic_id, terms, weights, sentences):
        self.topic_id = topic_id
        self.terms = terms
        self.weights = weights
        self.sentences = sentences

    def __str__(self):
        if self.sentences is None or len(self.sentences) == 0:
            return 'topic does not have any sentences'
        text = str()
        
        for t in self.terms:
            text += '{:s},'.format(t)
        text += '\n'
        
        for w in self.weights:
            text += '{:5.4f},'.format(w)
        text += '\n'
        for sentence in self.sentences:
            text += sentence[2] + ' '
        return text





class inference_summary(object):
    '''
    Generates summaries for a set of documents given a topic model.
    
    Parameters
    ----------
    model: TopicModel
        a TopicModel object trained on a corpus of documents
      
    num_dominant_topics: int, default: 5
        The number of dominant topics - corresponds to the
        number of summaries that are generated.

    number_of_sentences: int, default: 5
        The number of sentences per summary. 
        
    Attributes
    ----------
    summary_data: dictionary

    
    '''
    
    def __init__(self, lda, dictionary, num_dominant_topics=5, number_of_sentences=5):
        # the bigramizer should be the same object that was trained in TopicModel
        self.num_dominant_topics = num_dominant_topics
        self.number_of_sentences= number_of_sentences
        self.lda = lda
        self.dictionary = dictionary
    
    
    def summarize(self, documents):

        tokens = [tokenizer(document) for document in documents]
        #tokens = [self.bigramizer[tkn] for tkn in tokens]
        corpus = [self.dictionary.doc2bow(tkn) for tkn in tokens]
        self.dominant_topic = self.getDominantTopics(corpus)
        self.dominant_topic_ids = list(self.dominant_topic.keys())
        self.sentence_groups = self.splitIntoSentences(documents)
        self.distributions = self.getSentenceDistributions()
        self.topic_clusters = self.topic_clustering()
        self.summary_data = self.summarise_textrank()
            
    
    def getDominantTopics(self, corpus):
    
        # get topic weight matrix using lda.inference
        # the matrix has dimensions (num documents) x (num topics)
        inference = self.lda.inference(corpus)
        inference = inference[0] # the inference is a tuple, need the first term
        num_topics = self.lda.num_topics
        
        # find dominant topics across documents (vertical sum)
        column_sum_of_weights = np.sum(inference, axis=0)
        sorted_weight_indices = np.argsort(column_sum_of_weights)
        sorted_weight = np.sort(column_sum_of_weights)
        idx = np.arange(num_topics - self.num_dominant_topics, num_topics)
        dominant_topic_ids = sorted_weight_indices[idx]
        dominant_topic_weight = sorted_weight[idx]
        # the dominant_topic_ids store the ids in descending order of dominance
        dominant_topic_ids = dominant_topic_ids[::-1]
        
        return dict(zip(dominant_topic_ids, dominant_topic_weight))

    
    def splitIntoSentences(self, documents, MIN_SENTENCE_LENGTH = 8, MAX_SENTENCE_LENGTH = 25):
        sentence_groups = list()
        for document in documents:
            sentences = document.split(".")
            sentence_group = list()
            for k, sentence in enumerate(sentences):
                length = len(sentence.split())
                if (length > MIN_SENTENCE_LENGTH and length < MAX_SENTENCE_LENGTH):
                    sentence_group.append((k, sentence))
            sentence_groups.append(sentence_group)
        return sentence_groups
    
    
    def getSentenceDistributions(self):
        # computes topic distributions for each sentence
        # output: list of lists
        # each list corresponds to a document and stores a tuple per sentence
        # the 1st element is the sentence number in the group
        # the 2nd element is a tuple of (topic_id, weight)
        distributions = list()
        get_bow = self.dictionary.doc2bow
        get_document_topics = self.lda.get_document_topics
        for sentences in self.sentence_groups:
            sentence_distributions = list()
            for k, sentence in sentences:
                tkns = tokenizer(sentence)
                if tkns is None:
                    continue
                bow = get_bow(tkns)
                dist = get_document_topics(bow)
                # this is to get list of dominant indices in decreasing order
                #dist.sort(key=lambda x: x[1], reverse=True)
                #dist = [d[0] for d in dist]
                #
                # this is to get the dominant index only (not a list)
                try:
                    dist = max(dist, key=lambda x: x[1])
                except ValueError as  ve:
                    continue
                sentence_distributions.append((k, dist))
            distributions.append(sentence_distributions)
        return distributions
   

    def topic_clustering(self):

        """
        """
        topic_clusters = defaultdict(set)
        for distribution in self.distributions:
            for each_sent in distribution:
                if  each_sent[1][0] in self.dominant_topic_ids:
                    for sentence in self.sentence_groups:
                        for each in sentence:
                            if each[0] == each_sent[0]:
                                sen = each
                    topic_clusters[each_sent[1][0]].add(sen)
        return topic_clusters


    def summarise_textrank(self):
        
        topic_summary = defaultdict()
        for each_topic,sent  in self.topic_clusters.items():
            text_id = {x[1]:x[0] for x in sent}
            text = [x[1] for x in sent]
            if len(text)>1:
                summary = summarization(".".join(text))
            else:
                summary = text[0]
            topic_summary[each_topic] = summary
        return set(topic_summary.values())



    

