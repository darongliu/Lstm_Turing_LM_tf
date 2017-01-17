"""parsing text file"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import pickle

class data: 
    def __init__(self, data_dir, batch_size, min_seq_length, max_seq_length, min_count):
        self.train_file = os.path.join(data_dir, 'train.txt' )
        self.valid_file = os.path.join(data_dir, 'valid.txt' )
        self.test_file  = os.path.join(data_dir, 'test.txt'  )
        self.vocab_file = os.path.join(data_dir, 'data.vocab')

        self.train_tensor_file = os.path.join(data_dir, 'train.tensor')
        self.valid_tensor_file = os.path.join(data_dir, 'valid.tensor')
        self.test_tensor_file  = os.path.join(data_dir, 'test.tensor' )

        self.batch_size = batch_size
        self.min_count  = min_count
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length

        if not (os.path.exists(vocab_file)):
            print('vocab_file do not exist. Establishing vocab_file...')
            self.vocab_to_id = self.establish_vocab()
        else :
            self.vocab_to_id = pickle.load(self.vocab_file)

    def load(mode):
    if mode == 'train' then
        data = self.read_data(self.train_tensor_file)
    elif mode == 'valid' then
        data = self.read_data(self.train_tensor_file)
    elif mode == 'test' then
        data = self.read_data(self.train_tensor_file)
    else


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




        


