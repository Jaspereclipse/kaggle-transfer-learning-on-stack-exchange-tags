#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import os.path as osp
import numpy as np
import re
from bs4 import BeautifulSoup as bs
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import FreqDist
import cPickle as pkl

def combine(train_dict, test_dict):
    """Combine a list of csv files into data frame
    :param train_dict: topic-file dictionary for training set
    :param test_dict: topic-file dictionary for testing set
    :return: a combined data frame
    """
    df_train = _load_data(train_dict, train=True, offset=0)
    df_test = _load_data(test_dict, train=False, offset=len(train_dict))
    df_concat = pd.concat([df_train, df_test]).reset_index()

    return df_concat

def _load_data(dict_, train=True, offset=0):
    """Load csv files into a data frame; helper functions for combine()
    :param dict_: topic-file dictionary
    :return: a combined data frame
    """
    df_list = []
    for i, k in enumerate(dict_.keys()):
        df_local = pd.read_csv(dict_[k], encoding='utf-8')
        df_local['topic'] = i + offset
        df_list.append(df_local)
    df = pd.concat(df_list)
    df['qid'] = df['topic'].astype(str) + "-" + df['id'].astype(str)
    df.drop(['id', 'topic'], 1, inplace=True)
    if not train:
        df['tags'] = np.NaN
    df.reset_index(inplace=True)
    print "Loaded data."
    return df

def preprocess(data):
    """preprocess combined data frame:
        1. remove html tags;
        2. combine question title and content;
        3. remove spaces and new lines;
    :param data: combined data frame ['index', 'content', 'qid', 'tags']
    :return: cleaned data frame
    """
    removeHtmlTags = lambda x: bs(x, "html.parser").text
    data['cleaned_content'] = data['content'].apply(removeHtmlTags)
    print "Removed html tags."

    tmp = data['title'] + " " + data['cleaned_content']
    print "Combined question title and content."

    clean = lambda x: re.sub("[ ]+", " ", x.strip("\n"))
    data['combined_text'] = tmp.apply(clean)
    print "Removed extra spaces and new lines."

    keeps = ['qid', 'combined_text', 'tags']
    return data[keeps]

def build_word_dict(data, save_path, capped=None):
    """build word dictionary from training corpus:
    :param data: preprocessed data frame
    :param cap: an integer defining N most frequent words to keep,
                the rest are marked '_UNK_'
    :return: word2id and word2freq dictionary
    """
    corpus = data['combined_text'].str.cat(sep=' ')
    corpus = re.sub("[ ]+", " ", corpus)
    words = word_tokenize(corpus, 'english')
    print "There are %d tokens in corpus."%len(words)
    del corpus
    if not capped:
        capped = len(set(words))
    word2id = {}
    word2freq = {}
    word2id['_UNK_'] = 0
    sum_ = 0
    for word, freq in FreqDist(words).most_common(capped):
        word2id[word] = len(word2id)
        word2freq[word] = freq
        sum_ += freq
    word2freq['_UNK_'] = len(words) - sum_
    word2id['_NULL_'] = len(word2id)
    word2freq['_NULL_'] = -1
    print "Generated word dictionary; number of words: %d"%len(word2id)
    assert osp.isdir(save_path), "Invalid save path!"
    word2id_file = osp.join(save_path, "word2id.pkl")
    word2freq_file = osp.join(save_path, "word2freq.pkl")
    with open(word2id_file, 'wb') as f:
        pkl.dump(word2id, f)
    with open(word2freq_file, 'wb') as f:
        pkl.dump(word2freq, f)
    print "Dumped to: %s (counts: %d)"%(word2id_file, capped)
    print "Dumped to: %s (counts: %d)"%(word2freq_file, capped)
    return word2id, word2freq

def build_dataset(data, vocab, save_path, filter_dict={'min': 3, 'max': 50}):
    """build dataset for doc2vec training
    :param data: preprocessed data
    :param vocab: word2id dictionary
    :param filter_dict: parameters to filter out sentences that are either too short or too long
    :return: training set, e.g. ["qid", "sentence", "encoding", "length"]
    """
    # meta functions
    ssplit = lambda row: sent_tokenize(row['combined_text'], "english")
    get_df = lambda sents, row: pd.DataFrame(zip([row["qid"]]*len(sents), sents), columns=["qid", "sentence"])
    sent2id = lambda sent: [vocab[w] if w in vocab else 0 for w in word_tokenize(sent, "english")]

    first = data.loc[0, :]
    df_sent = get_df(ssplit(first), first)
    nrow = len(data.index)
    disp = nrow // 10
    for idx in xrange(1, nrow):
        row = data.loc[idx, :]
        df_sent = df_sent.append(get_df(ssplit(row), row), ignore_index=True)
        if (idx+1) % disp == 0:
            print "Processed: %d"%(idx+1)
    df_sent["encoding"] = df_sent["sentence"].apply(sent2id)
    df_sent["length"] = df_sent["encoding"].apply(len)
    df_sent = df_sent.loc[df_sent["length"] > filter_dict['min'], :]
    df_sent = df_sent.loc[df_sent["length"] < filter_dict['max'], :]
    df_sent['id'] = df_sent.groupby("qid").grouper.group_info[0] # numeric id for each paragraph
    df_sent = df_sent[['id', 'qid', 'encoding']]
    sents_file = osp.join(save_path, "sentences.csv") #number of paragraphs
    df_sent.to_csv(sents_file, index=False, encoding='utf-8')
    print "Saved to: %s; (paragraphs: %d)"%(sents_file, len(set(df_sent['qid'])))
    return df_sent

class Feed(object):
    """data feed for doc2vec graph model"""
    def __init__(self, training_data_csv, vocab_pkl, vocab_count_pkl):
        self.training_data = pd.read_csv(training_data_csv)
        vocab = self.load_pickle(vocab_pkl) # words2id
        self.null_id = vocab['_NULL_']
        self.id2word = {v:k for k, v in vocab.iteritems()} # id2word
        word2freq = self.load_pickle(vocab_count_pkl) # word2freq
        id2freq = {k: word2freq[v] for k, v in self.id2word.iteritems()} # id2freq
        self.vocab_counts = [id2freq[k] for k in sorted(id2freq.keys())]
        self._epoch = 1
        self.nrow = len(self.training_data.index)
        self.cnt_row = 0
        toIntList = lambda s: map(int, re.sub("[\[\]]", "", s).split(','))
        self.training_data['encoding'] = self.training_data['encoding'].apply(toIntList)
        print "Initialized feed."

    def load_pickle(self, pkl_file):
        assert osp.exists(pkl_file), "file does not exist: %s" % pkl_file
        with open(pkl_file, "rb") as f:
            target = pkl.load(f)
        print "Loaded %s" % pkl_file
        return target

    def get_vocal_counts(self):
        return self.vocab_counts

    def get_vocab_size(self):
        return len(self.id2word)

    def get_num_paragraphs(self):
        return len(set(self.training_data['id']))

    def get_batch(self, batch_size, window):
        """generate batch for stochastic gradient descent
        :param data: training data frame
        :param batch_size: num of windows each batch; type: ndarray with shape=(1,)
        :param window: window width; type: ndarray with shape=(1,)
        :param null_id: padding id; always set to len(vocab)
        :return: batch token ids, paragraph ids and labels
        """
        batch_words = np.ndarray(shape=(batch_size, window), dtype=np.int32)
        batch_paragraphs = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        sample = self.training_data.sample(n=batch_size//5).reset_index()
        sum_ = sample['encoding'].apply(len).sum()
        assert sum_ >= batch_size, "Not enough data: %d (< %d)" %(sum_, batch_size)

        cnt = 0 # fill count
        row = 0

        while cnt < batch_size:
            buffer_ = [self.null_id] * window + sample.loc[row, 'encoding'] # add paddings
            pid = sample.loc[row, 'id']
            for i in xrange(len(buffer_) - window): # slide window across sentence
                if cnt < batch_size:
                    batch_words[cnt, :] = buffer_[i: i + window]
                    batch_paragraphs[cnt, 0] = pid
                    labels[cnt, 0] = buffer_[i + window]
                    cnt += 1
                else:
                    break
            row += 1 # proceed to next sentence
        self.cnt_row += row
        if self.cnt_row >= self.nrow:
            self._epoch += 1
            self.cnt_row = 0
        return batch_words, batch_paragraphs, labels, np.int32(self._epoch)

if __name__ == '__main__':

    # set up
    train_data_dir = '../data/train/'
    train_topics = ['biology', 'cooking', 'crypto', 'diy', 'robotics', 'travel']
    train_files = [osp.join(train_data_dir, t+'.csv') for t in train_topics]
    train_dict = dict(zip(train_topics, train_files))
    test_dict = {'physics': '../data/test/test.csv'}
    #
    # # load-transform-save
    df = combine(train_dict, test_dict)
    df = preprocess(df)
    # save_path = "../data/tmp/"
    # feed = Feed(osp.join(save_path, "sentences.csv"), osp.join(save_path, "word2id.pkl"), osp.join(save_path, "word2freq.pkl"))