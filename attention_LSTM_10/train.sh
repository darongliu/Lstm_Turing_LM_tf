#! /bin/bash
mode='train'
init_from=''../LSTM/model/rnn_size300-0
init_method='lstm'
word_vector_path='../data/word2vec/GoogleNews-vectors-negative300.bin'

data_dir='../data/PTB_data'
save='./model/'
model_result='./result'
#att_file

rnn_size=300
emb_size=300
num_layers=1

batch_size=20
max_seq_length=60
min_seq_length=3

max_epochs=40
dropout=0.7
max_grad_norm=5
#entropy_reg

learning_rate=0.1
decay_rate=0.8
learning_rate_decay_after=25
init_scale=0.1

gpu_id=1
print_every=100

#CUDA_VISIBLE_DEVICES=$gpu_id python ./run.py --mode $mode --word_vector_path $word_vector_path --data_dir $data_dir --save $save --model_result model_result --rnn_size $rnn_size --emb_size $emb_size --num_layers $num_layers --batch_size $batch_size --max_seq_length $max_seq_length --min_seq_length $min_seq_length --max_epochs $max_epochs --dropout $dropout --max_grad_norm $max_grad_norm --learning_rate $learning_rate --decay_rate $decay_rate --learning_rate_decay_after $learning_rate_decay_after --init_scale $init_scale --print_every $print_every
CUDA_VISIBLE_DEVICES=$gpu_id python ./run.py --mode $mode --init_from $init_from --init_method $init_method --word_vector_path $word_vector_path --data_dir $data_dir --save $save --model_result model_result --rnn_size $rnn_size --emb_size $emb_size --num_layers $num_layers --batch_size $batch_size --max_seq_length $max_seq_length --min_seq_length $min_seq_length --max_epochs $max_epochs --dropout $dropout --max_grad_norm $max_grad_norm --learning_rate $learning_rate --decay_rate $decay_rate --learning_rate_decay_after $learning_rate_decay_after --init_scale $init_scale --print_every $print_every
