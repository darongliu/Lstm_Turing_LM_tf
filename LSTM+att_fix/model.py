import tensorflow as tf
import numpy

def inference(input_x, embedding_dim, lstm_hidden_dim_1, vocab_size,
    lstm_hidden_dim_2=None, dropout=None, window_size=5) :
    """
    Args:
        input_x: 2D tensor batch_size X time_step
        embedding_dim: embedding dimension
        lstm_hidden_dim_1: the dimension of the hidden unit of the bottom lstm 
        lstm_hidden_dim_2(optional): the dimension of the hidden unit of the top lstm
        vocab_size: vocabulary size
        dropout(optional): dropout keep probability, it should be a placeholder
    Returns:
        logits: predict result
        pretrain_list: the variable that can be pretrianed by lstm
        output_linear_list: the last output linear layer(cannot pretrained bt lstm)
    """
    pretrain_list = []
    output_linear_list = []

    #embedding layer
    with tf.variable_scope('embedding'):
        emb = tf.get_variable("emb", [vocab_size, embedding_dim])
        #init_width = 0.5 / embedding_dim
        #emb = tf.Variable(tf.random_uniform([vocab_size, embedding_dim], -init_width, init_width), name="emb")
        input_emb = tf.nn.embedding_lookup(emb, input_x)
        input_emb = tf.nn.dropout(input_emb, dropout)

    # add embedding matrix to pretrain list
    pretrain_list.append(emb)

    #lstm1 layer
    with tf.variable_scope('recurrent_layer1'):
        cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_hidden_dim_1, state_is_tuple=True)
        if dropout is not None:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout)
        
        """
        initial_state_c = tf.get_variable('initial_state_c', [1, lstm_hidden_dim_1])
        initial_state_h = tf.get_variable('initial_state_h', [1, lstm_hidden_dim_1])
        initial_state_c_batch = tf.tile(initial_state_c, [tf.shape(input_x)[0], 1])
        initial_state_h_batch = tf.tile(initial_state_h, [tf.shape(input_x)[0], 1])
        initial_state = tf.nn.rnn_cell.LSTMStateTuple(initial_state_c_batch, initial_state_h_batch)
        """
        initial_state = cell.zero_state(tf.shape(input_x)[0], tf.float32)
        
        lstm1_outputs, final_state = tf.nn.dynamic_rnn(cell, input_emb, initial_state=initial_state)
        #lstm1_outputs: [batch_size, num_steps, state_size]

    # add LSTM variable to pretrain list
    with tf.variable_scope('recurrent_layer1') as vs:
        lstm1_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)
        pretrain_list = pretrain_list + lstm1_variables

    with tf.variable_scope('attention_layer'):
        initial_state = tf.zeros([tf.shape(input_x)[0],1,lstm_hidden_dim_1])
        memory = lstm1_outputs
        #memory: [batch_size, num_steps, state_size]
        W_key = tf.get_variable('W_key', [lstm_hidden_dim_1, lstm_hidden_dim_1])
        b_key = tf.get_variable('b_key', [lstm_hidden_dim_1], initializer=tf.constant_initializer(0.0))

        def step(_,time_step):
            """
            args:
                -: dummy input(for the reinput of the output)
                time_step: from 0 to the last time step
            """
            """
            current_hidden = memory[:,time_step+1,:]
            previous_hidden = memory[:,:time_step+1,:]
            """
            real_window_size = tf.minimum(window_size,time_step)
            current_hidden = tf.slice(memory, [0,time_step,0], [-1,1,-1])
            current_hidden = tf.squeeze(current_hidden, [1])
            previous_hidden = tf.slice(memory, [0,time_step-real_window_size,0], [-1,real_window_size,-1])
            previous_hidden = tf.concat(1,[initial_state,previous_hidden])
            #previous_hidden: [batch_size, num_steps, state_size]

            key = tf.matmul(current_hidden,W_key) + b_key
            key = tf.tanh(key)
            #key: [batch_size, state_size]

            key_as_matrix = tf.expand_dims(key,2)
            similarity = tf.batch_matmul(previous_hidden, key_as_matrix)
            similarity = tf.squeeze(similarity,[2])
            #similarity = [batch_size, num_steps]
            weight = tf.nn.softmax(similarity)

            weight_as_matrix = tf.expand_dims(weight,1)
            attention = tf.batch_matmul(weight_as_matrix, previous_hidden)
            attention = tf.squeeze(attention,[1])

            return attention

        time_step_sequence = tf.range(tf.shape(input_x)[1])
        #time_step_sequence = tf.to_int32(time_step_sequence)
        initializer = tf.zeros([tf.shape(input_x)[0], lstm_hidden_dim_1])

        att_outputs = tf.scan(step, time_step_sequence, initializer=initializer)
        att_outputs = tf.transpose(att_outputs, [1,0,2])
        att_outputs = tf.nn.dropout(att_outputs, dropout)

    with tf.variable_scope('merge_layer'):
        att_lstm_outputs = tf.concat(2,[lstm1_outputs,att_outputs])

    #output layer which is attached after lstm1 
    with tf.variable_scope('output_lstm1_att_linear'):
        W = tf.get_variable('W', [2*lstm_hidden_dim_1, vocab_size])
        b = tf.get_variable('b', [vocab_size], initializer=tf.constant_initializer(0.0))

        recurrent_outputs = tf.reshape(att_lstm_outputs,[-1,2*lstm_hidden_dim_1])
        logits = tf.matmul(recurrent_outputs, W) + b
        logits = tf.reshape(logits, [tf.shape(input_x)[0], tf.shape(input_x)[1], vocab_size])

    #dropout, pretrain
    #add pretrain_param, output_linear_param
    output_linear_list.append(W)
    output_linear_list.append(b)

    return logits, pretrain_list, output_linear_list

def loss(logits, labels, entropy=None, entropy_reg=0) :
    """
    args:
        logits: [batch_size, num_steps, vocab_size] dtype='float32'
        labels: [batch_size, num_steps] dtype='int'
    return :
        total_label_loss: the summation of the full tensor loss
        loss: for training
    """
    cross_entropy_result = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    total_label_loss = tf.reduce_sum(cross_entropy_result)
    #devide vocab size
    loss = tf.reduce_mean(cross_entropy_result)
    if entropy is not None:
        loss = loss + entropy_reg*entropy

    return total_label_loss, loss

def training(loss, learning_rate, grad_norm) :
    """
    args:
        loss
        learning_rate: it should be a placeholder
        grad_norm: max grad norm

    return :
        train_op
    """
    """
    # Create the gradient descent optimizer with the given learning rate.
    #optimizer = tf.train.AdamOptimizer(learning_rate)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    gvs = optimizer.compute_gradients(loss)
    capped_gvs = [(tf.clip_by_global_norm(grad, grad_norm), var) for grad, var in gvs]
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.apply_gradients(capped_gvs)
    """
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(zip(grads, tvars))

    return train_op
    




