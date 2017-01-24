from __future__ import print_function

import tensorflow as tf

def inference(input_x, embedding_dim, lstm_hidden_dim_1, lstm_hidden_dim_2=None) :
    """
    Args:
    input_x: 2D tensor batch_size X time_step
    embedding_dim: embedding dimension
    hidden_unit_list: list of the hidden unit size
    lstm_hidden_dim_1: the dimension of the hidden unit of the bottom lstm 
    lstm_hidden_dim_2(optional): the dimension of the hidden unit of the top lstm

    Returns:

    """

    with tf.name_scope('embedding'):
        init_width = 0.5 / embedding_dim
        emb = tf.Variable(
            tf.random_uniform(
            [vocab_size, embedding_dim], -init_width, init_width),
            name="emb")
        input_emb = tf.nn.embedding_lookup(emb, input_x)

    with tf.name_scope('recurrent_layer1'):
        cell = tf.nn.rnn_cell.LSTMCell(lstm_hidden_dim_1, state_is_tuple=True)
        initial_state_vector = tf.get_variable('initial_state_vector', [1, lstm_hidden_dim_1])
        initial_state = tf.tile(initial_state_vector, [tf.shape(input_x)[0], 1])
        lstm1_outputs, final_state = tf.nn.dynamic_rnn(cell, input_emb, initial_state=initial_state)
        #lstm1_outputs: [batch_size, num_steps, state_size]

    with tf.name_scope('attention_layer'):
    #using scan
    #TODO
        memory = tf.concat([initial_state,lstm1_outputs],1)
        #memory: [batch_size, num_steps, state_size]
        W_key = tf.get_variable('W_key', [lstm_hidden_dim_1, lstm_hidden_dim_1])
        b_key = tf.get_variable('b_key', [lstm_hidden_dim_1], initializer=tf.constant_initializer(0.0))

        W_memory_selection_1 = tf.get_variable('W_memory_selection_1', [lstm_hidden_dim_1, lstm_hidden_dim_1])
        b_memory_selection_1 = tf.get_variable('b_memory_selection_1', [lstm_hidden_dim_1], initializer=tf.constant_initializer(0.0))

        W_memory_selection_2 = tf.get_variable('W_memory_selection_2', [lstm_hidden_dim_1, lstm_hidden_dim_1])
        b_memory_selection_2 = tf.get_variable('b_memory_selection_2', [lstm_hidden_dim_1], initializer=tf.constant_initializer(0.0))

        def step(time_step):
            """
            args:
            timestep:from 0 to the last time step
            """
            current_hidden = memory[:,time_step+1,:]
            previous_hidden = memory[:,:time_step+1,:]
            #previous_hidden: [batch_size, num_steps, state_size]

            memory_selection_1 = tf.sigmoid(tf.matmul(current_hidden,W_memory_selection_1) + b_memory_selection_1)
            memory_selection_2 = tf.sigmoid(tf.matmul(current_hidden,W_memory_selection_2) + b_memory_selection_2)
            #memory selection: [batch_size, state_size]
            key = tf.matmul(current_hidden,W_key) + b_key
            #key: [batch_size, state_size]

            memory_selection_1_as_matrices = tf.expand_dims(memory_selection_1,1)
            similarity_temp = tf.multiply(previous_hidden,memory_selection_1_as_matrices) 
            key_as_matrices = tf.expand_dims(key,2)
            similarity = tf.batch_matmul(similarity_temp, key_as_matrices)
            #similarity = [batch_size, num_steps]
            weight = tf.nn.softmax(similarity)
            weight_pad = 

            




        rnn_outputs, final_states = \
            tf.scan(lambda a, x: cell(x, a[1]),
                    tf.transpose(rnn_inputs, [1,0,2]),
                    initializer=(tf.zeros([batch_size, state_size]), init_state))
        att_outputs




    with th.name_scope('merge_layer'):
        if merge_mode == 'concat':
            att_lstm_outputs = tf.concat([lstm1_outputs,att_outputs],2)
        else if merge_mode == 'matrix':
            merge_matrix = tf.Variable(
                tf.random_uniform(
                [lstm_hidden_dim_1, lstm_hidden_dim_1], -1, 1),
                name="merge_matrix")
            att_lstm_outputs = tf.matmul(lstm1_outputs,att_outputs) + lstm1_outputs
        else if merge_mode == 'alpha':
            merge_alpha = tf.Variable(
                1,
                name="merge_alpha")
            att_lstm_outputs = merge_alpha*lstm1_outputs + 
                                (1-merge_alpha)*att_outputs
        else:
            print("unknown merge mode, using concat")
            att_lstm_outputs = tf.concat([lstm1_outputs,att_outputs],2)

    with tf.name_scope('recurrent_layer2'):
        cell = tf.nn.rnn_cell.LSTMCell(lstm_hidden_dim_2, state_is_tuple=True)
        initial_state_vector = tf.get_variable('initial_state_vector', [1, lstm_hidden_dim_2])
        initial_state = tf.tile(initial_state_vector, [tf.shape(input_x)[0], 1])
        lstm2_outputs, final_state = tf.nn.dynamic_rnn(cell, att_lstm_outputs, initial_state=initial_state)
        att_lstm_outputs = lstm2_outputs

    with tf.name_scope('softmax'):
        W = tf.get_variable('W', [lstm_hidden_dim_2, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))

        logits = tf.matmul(att_lstm_outputs, W) + b

    return logits

def loss() :
    total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y_reshaped))

def training() :

def evaluation() :
