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

        def step(_,time_step):
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
            entropy_temp = -tf.multiply(weight,tf.log(weight))
            entropy = tf.reduce_sum(entropy_temp,1)

            weight_pad_length = tf.shape(input_x)[1] - time_step - 1
            weight_pad = tf.concat([weight,tf.zeros([tf.shape(input_x)[0],weight_pad_length])],1) #for weight visualization

            memory_selection_2_as_matrices = tf.expand_dims(memory_selection_2,1)
            attention_temp = tf.multiply(previous_hidden,memory_selection_2_as_matrices) 
            weight_as_matrics = tf.expand_dims(weight,1)
            attention = tf.batch_matmul(weight_as_matrics, previous_hidden)
            attention = tf.squeeze(attention,[1])

            return attention, weight_pad, entropy

        time_step_sequence = tf.range(0,tf.shape(input_x)[2])
        initializer = [tf.zeros([tf.shape(input_x)[0], lstm_hidden_dim_1]),
                       tf.zeros_like(input_x),
                       tf.zeros([tf.shape(input_x)[0],])]

        att_outputs, weight_outputs, entropy_outputs = \
            tf.scan(step,time_step_sequence,initializer=initializer)

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

    #with tf.name_scope('entropy_loss') :
    #dropout, no num_classes

    return logits

def loss(logits, labels, entropy=None, entropy_reg=0) :
    label_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y_reshaped))

def training(loss, learning_rate) :
    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar('loss', loss)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def evaluation(logits, labels) :
