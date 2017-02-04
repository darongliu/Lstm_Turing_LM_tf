from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import argparse

import numpy as np
import tensorflow as tf

import reader
from model import *
def parsing_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train',
                        help='train or test')
    parser.add_argument('--init_from', type=str, default='',
                        help='init model path')
    parser.add_argument('--init_method', type=str, default='lstm',
                        help='lstm/att init from lstm or full model')

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
    parser.add_argument('--entropy_reg', type=float, default=0.1,
                        help='entropy regulizar')

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
    
def run_epoch_training(sess, all_op, data, lr):
    start_time = time.time()
    nbatch = data.get_batch_number()
    total_words_num = 0
    total_cost = 0

    fetches = {}
    fetches["train_op"] = all_op["train_op"]
    fetches['label_loss'] = all_op['label_loss_op']

    for idx in range(nbatch):
        x, y = data.get_data(idx)

        feed_dict = {all_op['input_data']:x, all_op['targets']:y, all_op['learning_rate_op']:lr}

        result = sess.run(fetches,feed_dict=feed_dict)
        total_cost += result['label_loss']
        total_words_num += x.size

        print idx+1, '/', 'nbatch: ', 'perplexity: ', np.exp(result['label_loss']/x.size)

    total_perplexity = np.exp(total_cost/total_words_num)
    print 'training perplexity in this epoch: ', total_perplexity
    print 'epoch training time', (time.time() - start_time)

    return total_perplexity

def evaluating(sess, all_op, data):
    nbatch = data.get_batch_number()
    total_words_num = 0
    total_cost = 0
    for idx in range(nbatch):
        x, y = data.get_data(idx)

        feed_dict = {all_op['input_data']:x, all_op['targets']:y}
        label_loss_op = all_op['label_loss_op']

        cost = sess.run(label_loss_op,feed_dict=feed_dict)
        total_cost += cost
        total_words_num += x.size

    perplexity = np.exp(total_cost/total_words_num)
    print 'evaluating perplexity: ', perplexity

    return perplexity

def train(args):
    #read data
    train_data = reader.data(data_dir=args.data_dir, 
                             batch_size=args.batch_size, 
                             min_seq_length=args.min_seq_length, 
                             max_seq_length=args.max_seq_length,
                             min_count=args.min_count)
    train_data.load('train')
    valid_data = reader.data(data_dir=args.data_dir, 
                             batch_size=args.batch_size, 
                             min_seq_length=args.min_seq_length, 
                             max_seq_length=args.max_seq_length,
                             min_count=args.min_count)
    valid_data.load('valid')
    test_data = reader.data(data_dir=args.data_dir, 
                             batch_size=args.batch_size, 
                             min_seq_length=args.min_seq_length, 
                             max_seq_length=args.max_seq_length,
                             min_count=args.min_count)
    test_data.load('test')

    #load model
    if args.init_from:
        if is not os.path.isfile(args.init_from):
            print 'init file not found'
            os.exit()

    #training ops
    input_data = tf.placeholder(tf.int32, [None, None])
    targets    = tf.placeholder(tf.int32, [None, None])
    learning_rate_op = tf.placeholder(tf.float32, [])
    #build model
    vocab_size=train_data.vocab_size
    predict, pretrain_param, output_linear_param = model.inference(input_data, 
                                                                   embedding_dim=args.emb_size
                                                                   lstm_hidden_dim_1=args.rnn_size
                                                                   vocab_size=vocab_size)

    label_loss_op, total_loss_op = model.loss(targets, predict, entropy=True, entropy_reg=args.entropy_reg)
    train_op = model.training(total_loss_op, learning_rate_op)

    all_op = {'input_data':input_data, 
              'targets':targets
              'learning_rate_op':learning_rate_op
              'label_loss_op':label_loss_op
              'train_op':train_op}


    #pretrain
    if args.init_method == 'lstm':
        lstm_linear_w = tf.variable(tf.random_uniform([args.rnn_size, vocab_size], 0, 0),name='output_lstm_linear/w')
        lstm_linear_b = tf.variable(tf.random_uniform([vocab_size], 0, 0),name='output_lstm_linear/b')
        init_att_w = output_linear_param['w'].assign(tf.concat([lstm_linear_w,lstm_linear_w],0))
        init_att_b = output_linear_param['b'].assign(lstm_linear_b)
        saver = tf.train.Saver(pretrain_param)
    else:
        saver = tf.train.Saver()
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)
        if args.init_method == 'lstm':
            saver.restore(sess, args.init_from)
            sess.rum(init_att_w)
            sess.rum(init_att_b)
        else:
            saver.restore(sess, args.init_from)

        #training
        best_val_perplexity = np.inf

        for i in range(args.max_epochs):
            lr_decay = args.decay_rate ** max(i + 1 - args.learning_rate_decay_after, 0.0)
            learning_rate = args.learning_rate * lr_decay
            print("Epoch: %d Learning rate: %.3f" % (i + 1, learning_rate))

            #training
            training_perplexity = run_epoch_training(sess, all_op, train_data, learning_rate)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, training_perplexity))

            #validation
            val_perplexity = evaluating(sess, all_op, valid_data)
            print("Epoch: %d validation Perplexity: %.3f" % (i + 1, val_perplexity))

            #peeking testing
            test_perplexity = evaluating(sess, all_op, test_data)
            print("Epoch: %d peeking testing Perplexity: %.3f" % (i + 1, test_perplexity))

            if val_perplexity < best_val_perplexity :
                best_val_perplexity = val_perplexity
                #save
                sv.saver.save(session, args.save, global_step=sv.global_step)
                #TODO

def test(args):
    test_data = reader.data(data_dir=args.data_dir, 
                             batch_size=args.batch_size, 
                             min_seq_length=args.min_seq_length, 
                             max_seq_length=args.max_seq_length)
    test_data.load('test')
    #build model
    input_data = tf.placeholder(tf.int32, [None, None])
    targets    = tf.placeholder(tf.int32, [None, None])
    learning_rate_op = tf.placeholder(tf.float32, [])
    #build model
    vocab_size=test_data.vocab_size
    predict, pretrain_param, output_linear_param = model.inference(input_data, 
                                                                   embedding_dim=args.emb_size
                                                                   lstm_hidden_dim_1=args.rnn_size
                                                                   vocab_size=vocab_size)

    label_loss_op, total_loss_op = model.loss(targets, predict, entropy=True, entropy_reg=args.entropy_reg)
    train_op = model.training(total_loss_op, learning_rate_op)

    all_op = {'input_data':input_data, 
              'targets':targets
              'learning_rate_op':learning_rate_op
              'label_loss_op':label_loss_op
              'train_op':train_op}

    #load model
    saver = tf.train.Saver()
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, args.init_from)

        test_perplexity = evaluating(sess, all_op, test_data)
        print ("Testing Perplexity: %.3f" % (test_perplexity))









#read data
if __name__ == "__main__":
    args = parsing_args()
    if args.mode == 'train':
        train(args)
    else :
        test(args)


