from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import argparse

import numpy as np
import tensorflow as tf

import reader
def parsing_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train',
                        help='train or test')

    parser.add_argument('--data_dir', type=str, default='data/',
                        help='data directory containing train valid test data')
    parser.add_argument('--log_file', type=str, default='logs',
                        help='directory containing tensorboard logs')
    parser.add_argument('--save', type=str, default='save',
                        help='directory to store checkpointed models')
    parser.add_argument('--load', type=str, default=None,
                        help="continue training from saved model at this path. Path must contain files saved by previous training process")
    parser.add_argument('--att_file', type=str, default='save',
                        help='file storing attention weights for analysis')

    parser.add_argument('--rnn_size', type=int, default=200,
                        help='size of LSTM internal state')
    parser.add_argument('--emb_size', type=int, default=200,
                        help='word embedding size')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in the RNN')

    parser.add_argument('--batch_size', type=int, default=20,
                        help='minibatch size')
    parser.add_argument('--max_seq_length', type=int, default=40,
                        help='max number of timesteps to unroll during BPTT')
    parser.add_argument('--min_seq_length', type=int, default=15,
                        help='min number of timesteps to unroll during BPTT')

    parser.add_argument('--max_epochs', type=int, default=50,
    help='number of full passes through the training data')
    parser.add_argument('--dropout', type=int, default=0,
                        help='dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
    parser.add_argument('--max_grad_norm', type=float, default=5.,
                        help='clip gradients at this value')

    parser.add_argument('--learning_rate', type=float, default=1.0,
                        help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.5,
                        help='decay rate')
    parser.add_argument('--learning_rate_decay_after', type=int, default=4,
                        help='in number of epochs, when to start decaying the learning rate')

    parser.add_argument('--gpu_id', type=float, default=0.666,
                        help='% of gpu memory to be allocated to this process. Default is 66.6%')
    parser.add_argument('--print_every', type=int, default=200,
                        help='how many steps/minibatches between printing out the loss')
    parser.add_argument('--seed', type=int, default=40,
                        help='random number generator seed')

    args = parser.parse_args()
    return args

def run_minibatch(session):
    
def train(args):
    #read data
    train_data = reader.data(data_dir=args.data_dir, 
                             batch_size=args.batch_size, 
                             min_seq_length=args.min_seq_length, 
                             max_seq_length=args.max_seq_length)
    train_data.load('train')
    valid_data = reader.data(data_dir=args.data_dir, 
                             batch_size=args.batch_size, 
                             min_seq_length=args.min_seq_length, 
                             max_seq_length=args.max_seq_length)
    valid_data.load('valid')

    #load model
    if args.init_from is not None:
    #pretrain
    #training



def test(args):
    test_data = reader.data(data_dir=args.data_dir, 
                             batch_size=args.batch_size, 
                             min_seq_length=args.min_seq_length, 
                             max_seq_length=args.max_seq_length)
    test_data.load('train')

#read data
if __name__ == "__main__":
    args = parsing_args()
    if args.mode == 'train':
        train(args)
    else :
        test(args)


