# coding=utf-8

import math
import numpy as np

class TextRank:
    """ This class is the implementation of TextRank proposed by following paper.
    TextRank is a graph-based summarization model that use PageRank like algorithm.
    This class does not perform tokenization and normalization of texts, so you should preprocess your data beforehand.

    Original Paper: TextRank: Bringing Order into Texts, Mihalcea+, EMNLP 2004.
    https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf
    """

    def __init__(self, conv_thr=0.0001, damping_factor=0.85, stopwords_path="./smart_common_words.txt"):
        """ Initializer for TextRank Class
        Params:
            conv_thr(float): convergence threshold on TextRank
            damping_factor(float): damping factor in TextRank
            stopwords_path(str): stopword list (this file should be represent 1 word per line)
        """
        self.CONV_THR = conv_thr
        self.D_FACTOR = damping_factor
        try:
            with open(stopwords_path) as f:
                self.stopwords_list = [w for w in f.read().strip().split("\n")]
        except IOError:
            self.stopwords_list = []
 
    def __calc_similarity(self):
        """ Calculate similarity based on similarity function in Sec. 4.1.
        """
        def __sim_func(i, j):
            tokens1, tokens2 = self.sents[i], self.sents[j]
            overlap = len(set(tokens1) & set(tokens2))

            if i==j or overlap==0:
                return 0.0
            return overlap / (math.log(len(tokens1)) + math.log(len(tokens2)))
        
        N = len(self.sents)
        _sim_mat = np.asarray([[__sim_func(i, j) for j in range(N)] for i in range(N)], dtype=np.float32)
        # normalize (sum of row equals 1)
        self.sim_mat = np.asarray([_sim_mat[i]/sum(_sim_mat[i]) if sum(_sim_mat[i]!=0.0) else _sim_mat[i] for i in range(N)], dtype=np.float32)
  
    def __calc_diff(self, vec1, vec2):
        """ Calculate difference between vec1 and vec2.
        Params:
            vec1(numpy.array): numpy array that we want to calculate difference
            vec2(numpy.array): numpy array that we want to calculate difference
        """
        return sum(np.fabs(vec1 - vec2))

    def set_sentences(self, sents):
        """ Set sentences that you want to calculate TextRank score
        Params:
            sents(list): list of sentences, a sentence should be represented on list of tokens
        """
        self.sents = sents

    def run(self, debug=False):
        """ Run iteration to calculate TextRank score.
        Params:
            debug(boolean): if debug equals True, #_of_iter and diff will be presented at each iter.
        """
        self.__calc_similarity()
        
        diff = 1
        N = len(self.sents)
        tr = np.asarray([1.0 / N for _ in range(N)], dtype=np.float32)
        iter_count = 0
        while diff > self.CONV_THR:
            next_tr = (1.0 - self.D_FACTOR) + self.D_FACTOR * self.sim_mat.T.dot(tr)
            diff = self.__calc_diff(next_tr, tr)
            tr = next_tr

            if debug:
                iter_count += 1
                #print("Iteration {0}: diff={1}".format(iter_count, diff))

        self.text_rank = tr

def summarization(text):
    tr = TextRank()
    #text = 'Ob die APA die Qualität auch mit weniger Mitarbeitern aufrechterhalten könne?.Hauptkritikpunkt der Belegschaft ist, dass Sparmaßnahmen angekündigt würden, obwohl die APA seit Jahren Gewinne schreibe..Persönlich habe er außerdem nur eine geringe Beteiligung an der Betriebsversammlung wahrgenommen..Die als Genossenschaft organisierte Agentur gehört zu 45 Prozent dem ORF, den Rest teilen sich 13 Tageszeitungen..Für 2016 steige das Personalbudget sogar, allerdings würden automatische Gehaltserhöhungen die Kosten für den einzelnen Mitarbeiter steigen lassen.. Der Betriebsrat glaubt, dass man die Kosten einfach unbegrenzt weiterlaufen lassen kann, so Kropsch.'
    sentences = text.split(".")
    sents_toks = [sent.split(" ") for sent in sentences if sent != ""]
    tr.set_sentences(sents_toks)
    tr.run(debug=True)
    ids = [i for i, _ in sorted(enumerate(tr.text_rank), key=lambda x: x[1], reverse=True)[:5]]
    summary = []
    max_rank =0
    for i in ids:
        if tr.text_rank[i] > max_rank:
            summary = sentences[i]
            max_rank = tr.text_rank[i]
    return summary
    print("\n".join(["["+str(i)+":"+str(tr.text_rank[i])+"]"+" "+sentences[i] for i in ids]))


