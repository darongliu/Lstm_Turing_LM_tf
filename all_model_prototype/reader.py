""" 
modifier: Da-Rong Liu
Date: 2/01/2017
In this implementation, sentences of the same length are put into one bucket.
This helps to avoid doing padding.

Refenrence: https://github.com/ketranm/RMN/blob/master/text/TextProcessor.lua
"""

import collections
import os
import sys
import pickle
import numpy as np
from random import shuffle
import gensim

class data: 
    def __init__(self, data_dir, batch_size=100, min_seq_length=10, max_seq_length=50, min_count=2):
        self.train_file = os.path.join(data_dir, 'train.txt' )
        self.valid_file = os.path.join(data_dir, 'valid.txt' )
        self.test_file  = os.path.join(data_dir, 'test.txt'  )
        self.vocab_file = os.path.join(data_dir, 'data.vocab')

        self.batch_size = batch_size
        self.min_count  = min_count
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length

        self.x = self.y = self.nbatch = self.all_tensor_data = None

        if not (os.path.exists(vocab_file)):
            print('vocab_file do not exist. Establishing vocab_file...')
            self.vocab_to_id = self.establish_vocab()
        else :
            print('loading vocab_file...')
            self.vocab_to_id = pickle.load(self.vocab_file)
        self.vocab_size = len(self.vocab_to_id)

    def load(mode):
    if mode == 'train': 
        print('loading train text file...')
        data = self.read_data(self.train_file)
    elif mode == 'valid': 
        print('loading valid text file...')
        data = self.read_data(self.valid_file)
    elif mode == 'test': 
        print('loading test text file...')
        data = self.read_data(self.test_file)
    else:
        print('mode must be train, valid, or test...')
        sys.exit()

    buckets = self.create_buckets(self.min_seq_length, self.max_seq_length, data)
    self.all_tensor_data = self.text_to_tensor(buckets, self.vocab_to_id, self.min_seq_length, self.max_seq_length, data)
    self.x, self.y, self.nbatch = self.generate_batch(self.batch_size, self.all_tensor_data)

    def get_data(index) :
        if not self.x :
            print "still not load data..."
            return None
        else :
            return [self.x[index], self.y[index]]

    def get_batch_number() :
        if not self.nbatch :
            print "still not load data..."
            return None
        return self.nbatch

    def shuffling_data():
        if not self.x :
            print "still not load data..."
        else :
            print "shuffling data..."
            self.x, self.y, self.nstep = self.generate_batch(self.batch_size, self.all_tensor_data)

    """ -------STATIC METHOD------- """     
    def establish_vocab():
        print('loading train text file...')
        train_data = self.read_data(self.train_file)

        all_words = []
        for sentence in train_data :
            if len(sentence) <= self.max_seq_length and len(sentence) >= self.min_seq_length :
                all_words = all_words + sentence.split()

        print('creating vocabulary mapping...')
        vocab_to_id = {}
        counter = collections.Counter(all_words) #counter: dict {vocab:times}
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0])) #sort by ascending
        for word, count in count_pairs :
            if count >= self.min_count :
                vocab_to_id[word] = len(vocab_to_id)

        special_words = {"<unk>"} #used for any unknown words
        for special_word in special_words:
            if special_word not in vocab_to_id :
                vocab_to_id[special_word] = len(vocab_to_id)

        #save vocab file
        pickle.dump(vocab_to_id, self.vocab_file)

        return vocab_to_id

    def read_data(filename):
        with open(filename, "r") as f:
            content = f.readlines()
        return content
        #add <s> and </s> at both end of the sentences
        #return ["<s> "+line+" </s>" for line in content]

    def create_buckets(min_length, max_length, data) :
        """
        count the number of each length of the sentences
        data: list of sentences
        buckets: dict {length: number}
        """
        buckets = {}
        for line in data :
            words = line.split()
            length = len(words)
            if length <= max_length and length >= min_length :
                if length is in buckets :
                    buckets[length] = buckets[length] + 1
                else :
                    buckets[length] = 1
        return buckets

    def text_to_tensor(buckets, vocab_to_id, min_length, max_length, data) :
        """
        transform text data to tensor format
        all_data: dict {length: the tensor of the length}
        """
        all_data = {}
        all_data_count = {}
        for length, sentence_count in buckets :
            all_data[length] = np.zeros([sentence_count,length])
            all_data_count[length] = 0

        for line in data :
            words = line.split()
            length = len(words)
            if length <= max_length and length >= min_length :
                count = 0
                for word in words :
                    if word is in vocab_to_id :
                        all_data[length][all_data_count[length]][count] = vocab_to_id[word]
                    else :
                        all_data[length][all_data_count[length]][count] = vocab_to_id["<unk>"]
                    count = count + 1

                all_data_count[length] = all_data_count[length] + 1

        return all_data

    def generate_batch(batch_size, all_tensor_data):
        """
        transform all tensor data into batch form
        """
        all_data = {}
        for length, tensor in all_tensor_data:
            all_data[length] = np.random.shuffle(all_tensor_data[length])

        all_batch = []
        for length, tensor in all_data:
            sentence_num = tensor.shape[0]
            batch_num    = sentence_num // batch_size
            remaining = sentence_num - batch_num*batch_size
            for i in range(batch_num) :
                all_batch.append(all_data[length][i*batch_size:(i+1)*batch_size,:])
            if remaining :
                all_batch.append(all_data[length][batch_num*batch_size:,:])

        all_shuffle_batch = shuffle(all_batch)

        x = [tensor[:,:-1] for tensor in all_shuffle_batch]
        y = [tensor[:,1:]  for tensor in all_shuffle_batch]
        num = len(all_shuffle_batch)

        return x, y, num

    def generate_word_embedding_matrix(vocab_to_id, path):
        """
        generate vocab lookup embedding matrix from pretrained word2vector
        args:
            vocab_to_id:
            path: model path
        return:
            embedding_matrix: pretrained word embedding matrix
        """
        model = gensim.models.Word2Vec.load_word2vec_format(path, binary=True)
        all_vocab_vector = []
        for word, idx in vocab_to_id:
            try:
                word_vector = model[word]
            except:
                word_vector = np.zeros([300],dtype='float32')
            all_vocab_vector.append(word_vector)

        embedding_matrix = np.concatenate9all_vocab_vector,axis=0)
        return embedding_matrix


