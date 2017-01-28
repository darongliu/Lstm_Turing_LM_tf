"""parsing text file"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import pickle
import numpy as np
from random import shuffle

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

        self.x = self.y = self.nstep = self.all_classified_data = None

        if not (os.path.exists(vocab_file)):
            print('vocab_file do not exist. Establishing vocab_file...')
            self.vocab_to_id = self.establish_vocab()
        else :
            self.vocab_to_id = pickle.load(self.vocab_file)

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
        print('mode can only be train, valid, or test')

    buckets = self.create_buckets(self.min_seq_length, self.max_seq_length,data)
    self.all_classified_data = self.generate_classified_data(buckets, self.vocab_to_id, self.min_seq_length, self.max_seq_length, data)
    self.x, self.y, self.nbatch = self.generate_batch()

    def get_data(index) :
        if not self.x :
            print "still not load data"
            return None
        else :
            return [self.x[index], self.y[index]]

    def get_batch_number() :
        if not self.nbatch :
            print "still not load data"
            return None
        return self.nbatch

    def shuffling_data():
        if not self.x :
            print "still not load data"
        else :
            print "shuffling data"
            self.x, self.y, self.nstep = self.generate_batch()

    """ -------STATIC METHOD------- """     
    def establish_vocab():
        print('loading train text file...')
        train_data = read_data(self.train_file)

        all_words = []
        for sentence in train_data :
            if len(sentence) <= self.max_seq_length and len(sentence) >= self.min_seq_length :
                all_words = all_words + sentence.split()

        print('creating vocabulary mapping...')
        vocab_to_id = {}
        counter = collections.Counter(all_words)
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        for word, count in count_pairs :
            if count >= self.min_count :
                vocab_to_id[word] = len(vocab_to_id)

        special_words = {"<s>", "</s>", "<unk>"}
        for special_word in special_words:
            if special_word not in vocab_to_id :
                vocab_to_id[special_word] = len(vocab_to_id)

        #save vocab file
        pickle.dump(vocab_to_id, self.vocab_file)

        return vocab_to_id

    def read_data(filename):
        with open(filename, "r") as f:
            content = f.readlines()
        return ["<s> "+line+" </s>" for line in content]

    def create_buckets(min_length, max_length, data) :
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

    def generate_classified_data(buckets, vocab_to_id, min_length, max_length, data) :
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

    def generate_batch():
        all_data = {}
        for length, tensor in self.all_classified_data :
            all_data[length] = np.random.shuffle(self.all_classified_data[length])

        all_batch = []
        for length, tensor in all_data:
            sentence_num = tensor.shape[0]
            batch_num    = sentence_num // self.batch_size
            remaining = sentence_num - batch_num*self.batch_size
            for i in range(batch_num) :
                all_batch.append(all_data[length][i*self.batch_size:(i+1)*self.batch_size,:])
            if remaining :
                all_batch.append(all_data[length][batch_num*self.batch_size:,:])

        all_shuffle_batch = shuffle(all_batch)

        x = [tensor[:,:-1] for tensor in all_shuffle_batch]
        y = [tensor[:,1:]  for tensor in all_shuffle_batch]
        num = len(all_shuffle_batch)

        return x, y, num
