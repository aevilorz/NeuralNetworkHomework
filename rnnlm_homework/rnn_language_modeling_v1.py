'''
Created on 2017年4月10日

@author: Aevil
'''
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn  # rnn stuff temporarily in contrib, moving back to code in TF 1.1
import os
import time
import math
import numpy as np
import utils 
import pickle

tf.set_random_seed(0)

class rnnlm(object):
    
    def __init__(self,
                 data_processor,
                 batch_size,
                 seq_size,
                 embed_size,
                 hid_size,
                 nb_layers,
                 learning_rate = 0.001,
                 pr_keep = 1.0                 
                 ):
        self.dp = data_processor
        self.batch_size = batch_size
        self.seq_size = seq_size
        self.embed_size = embed_size
        self.hid_size = hid_size
        self.nb_layers = nb_layers
        self.lr = learning_rate
        self.pr_keep = pr_keep
        
        self.vocab_size = self.dp.vocabsize
        return
    
    def setup_placeholders(self):
        self.ph_lr = tf.placeholder(tf.float32, name = 'ph_lr')
        self.ph_pkeep = tf.placeholder(tf.float32, name = 'ph_keep')
        self.ph_batchsize = tf.placeholder(tf.int32, name = 'ph_batchsize')
        ''' [batch_size, seq_size] '''
        self.ph_batch_X = tf.placeholder(tf.int32, [None, None], name ='ph_X')
        ''' [batch_size, seq_size] '''
        self.ph_batch_T = tf.placeholder(tf.int32, [None, None], name ='ph_T')
        ''' [batch_size] '''
        self.ph_batch_seqlen = tf.placeholder(tf.int32, [None], name = 'ph_Seqlen')
        ''' [batch_size, seq_size] '''
        self.ph_batch_mask = tf.placeholder(tf.float32, [None, None], name = 'ph_Mask')
        ''' [batch_size, hid_size * nb_layers] '''
        self.ph_batch_Hin = tf.placeholder(tf.float32, [None, self.hid_size * self.nb_layers], name = 'ph_Hin')
        return 
    
    def setup_embedding_graph(self, batch_X_idx):
        ''' 
        add word Embedding Matrix into the current graph
        Embedding Matrix: 
            id ->Matrix(vocab_size, embed_size)-> word embedding vector (rnn's input)
        '''
        word_embeds = tf.Variable(tf.random_normal([self.vocab_size, self.embed_size], name = 'Embeddings'))
        '''
        batch_X_idx: [batch_size, seq_size]
        batch_Xin:   [batch_size, seq_size, embed_size]
        '''
        batch_Xin = tf.nn.embedding_lookup(word_embeds, batch_X_idx, name = 'lookup_op')
        return batch_Xin
    
    def setup_onehot_graph(self, batch_X_idx):
        '''
        transfer id -> word onehot vector
        '''
        
        '''[batch_size, seq_size, vocab_size]'''
        batch_Xin = tf.one_hot(batch_X_idx, self.vocab_size, 1.0, 0.0, name = 'Onehots')
        return batch_Xin
    
    def setup_gru_graph(self, 
                        batch_Xin,
                        batch_Hin, 
                        batch_seqlen, 
                        pr_keep, 
                        ):
        '''
        add GRU Rnn Layers into the current graph
        GRU Rnn Layers: 
            (input, initial hidden state) -> (output, final hidden state)
        
        #layers: nb_layers
        #neurons of each layer : hid_size
        the dropout probability of each layer : pr_keep
        
        PS: batch_Xin has sequences with different length (<= seq_size),
            batch_seqlen specifies the actual length of each sequences,
            batch_gru_O has the sequences with padding by zeros.
        '''
        gru_cell = rnn.GRUCell(self.hid_size)
        gru_cell_dropped = rnn.DropoutWrapper(gru_cell, input_keep_prob = pr_keep)
        stacked_cells = rnn.MultiRNNCell([gru_cell_dropped] * self.nb_layers, state_is_tuple = False)
        stacked_cells_dropped = rnn.DropoutWrapper(stacked_cells, output_keep_prob = pr_keep)
        '''
        batch_Xin:     [batch_size, seq_size, vect_size] # vect_size = embed_size if using embeddings else vocab_size
        batch_Hin:     [batch_size, hid_size * nb_layers]
        batch_seqlen:  [batch_size]
        
        batch_gru_O:       [batch_size, seq_size, hid_size]
        batch_gru_final_h: [batch_size, hid_size * nb_layers]
        '''
        batch_gru_O, batch_gru_final_h = tf.nn.dynamic_rnn(stacked_cells_dropped, 
                                                           batch_Xin, 
                                                           sequence_length = batch_seqlen, 
                                                           initial_state = batch_Hin
                                                           )
        batch_gru_final_h = tf.identity(batch_gru_final_h, name = 'Ho')
        return batch_gru_O, batch_gru_final_h
    
    def setup_softmax_graph(self, batch_rnn_O):
        '''
        add Full Connected Softmax Layer into the current graph
        Full Connected Softmax Layer: 
            rnn layer's output ->Matrix(hid_size, vocab_size)-> class distribution
                        
        #neurons: vocab_size
        PS: padding outputs are OK, 
            in sequences:
            actual rnn output vector -> actual distribution vector
            padding rnn output vector -> padding distribution vector
        '''
        
        '''[batch_size * seq_size, hid_size]'''
        batch_gru_O_flatted = tf.reshape(batch_rnn_O, [-1, self.hid_size])
        '''[batch_size * seq_size, vocab_size]'''
        batch_logits_flatted = layers.linear(batch_gru_O_flatted, self.vocab_size)
        '''[batch_size * seq_size, vocab_size]'''
        batch_final_O_flatted = tf.nn.softmax(batch_logits_flatted)
        '''[batch_size * seq_size]'''
        batch_final_O_idx_flatted = tf.argmax(batch_final_O_flatted, axis = 1)
        '''[batch_size, seq_size]'''
        batch_final_O_idx = tf.reshape(batch_final_O_idx_flatted, [self.ph_batchsize, -1])
#         batch_final_O_idx = tf.reshape(batch_final_O_idx_flatted, [batch_rnn_O.shape[0], -1])
        
        batch_final_O_flatted = tf.identity(batch_final_O_flatted, name = 'distr_O_flatted')
        batch_final_O_idx = tf.identity(batch_final_O_idx, name = 'O')
        
        return batch_logits_flatted, batch_final_O_idx
    
    def setup_performances_graph(self, 
                                 batch_T, 
                                 batch_Mask, 
                                 batch_Seqlen, 
                                 batch_logits_flatted, 
                                 batch_O_idx
                                 ):
        '''
        add cross-entropy loss, accuracy computations into the graph
        cross-entropy:
            loss = sum( 1{T} * log(distr_O) ) / len(T)
            use softmax_cross_entropy_with_logits to solve numerical unstable problem
        accuracy:
            acc = sum( 1{T = O}) / len(T)
        
        use Mask, Seqlen to prune the padding outputs
        '''
        
        '''[batch_size, seq_size, vocab_size]'''
        batch_T_onehot = tf.one_hot(batch_T, self.vocab_size, 1.0, 0.0)
        '''[batch_size * seq_size, vocab_size]'''
        batch_T_1h_flatted = tf.reshape(batch_T_onehot, [-1, self.vocab_size])
        '''[batch_size * seq_size]'''
        batch_Wordloss_flatted = tf.nn.softmax_cross_entropy_with_logits(
            labels = batch_T_1h_flatted, 
            logits = batch_logits_flatted)
        '''[batch_size, seq_size]'''
        batch_Wordloss = tf.reshape(batch_Wordloss_flatted, [self.ph_batchsize, -1])
#         batch_Wordloss = tf.reshape(batch_Wordloss_flatted, [batch_T.shape[0], -1])
        batch_Wordloss = batch_Wordloss * batch_Mask
        '''[batch_size]'''
        batch_Seqloss = tf.reduce_sum(batch_Wordloss, axis = 1) / tf.cast(batch_Seqlen, tf.float32)
        '''[1]'''
        onebatch_loss =tf.reduce_mean(batch_Seqloss)
        onebatch_loss = tf.identity(onebatch_loss, name = 'batch_loss')
                
        '''[batch_size, seq_size]'''
        batch_WordEqualFlag = tf.cast(
            tf.equal(batch_T, tf.cast(batch_O_idx, tf.int32)),
            tf.float32
            )
        batch_WordEqualFlag = batch_WordEqualFlag * batch_Mask
        '''[1]'''
        onebatch_acc = tf.reduce_sum(batch_WordEqualFlag) \
                        / tf.cast(tf.reduce_sum(batch_Seqlen), tf.float32)
        onebatch_acc = tf.identity(onebatch_acc, name = 'batch_acc')
        
        ''' TensorFlow Summary Operations'''
        loss_summ = tf.summary.scalar('loss', onebatch_loss)
        acc_summ = tf.summary.scalar('acc', onebatch_acc)
        '''
        summary_op is a Tensor of string type(Serialization),
        rename it as 'summ_op
        '''
        summary_op = tf.summary.merge([loss_summ, acc_summ])
        summary_op = tf.identity(summary_op, 'summ_op')
        return batch_Wordloss, summary_op
    
    def setup_optimizer_graph(self, loss_node, lr_node):
        '''
        add training operation into the current graph
        
        @train_op   its default operation name is 'Adam', rename it as 'trainer'
                    it already exists in tf.collection['train_op']
        '''
        train_op = tf.train.AdamOptimizer(lr_node).minimize(loss_node, name = 'trainer')
        return train_op
    
    def setup_initialization_graph(self):
        '''
        add initialization operation into the current graph
        global_variables_initializer()
            seems not to be working in this function
            but works well in train_model(....)
        
        @init_op    its default operation name is 'init', can NOT rename it
                    it is Not in the tf.collections
        '''
        init_op = tf.global_variables_initializer()
#         tf.add_to_collection('init_ops', init_op)
        return init_op
        
    def setup_whole_graph_from_codes(self, is_embedding = True):
        '''
        import the graph by above functions as tf.default_graph() 
        '''
        tf.reset_default_graph()
        
        self.setup_placeholders()
        
        if is_embedding:
            batch_Xin = self.setup_embedding_graph(self.ph_batch_X)
        else:
            batch_Xin = self.setup_onehot_graph(self.ph_batch_X)
        
        batch_rnn_O, batch_rnn_final_H = self.setup_gru_graph(batch_Xin, 
                                                              self.ph_batch_Hin, 
                                                              self.ph_batch_seqlen, 
                                                              self.ph_pkeep)
        batch_logits_flatted, batch_O_idx = self.setup_softmax_graph(batch_rnn_O)
        batch_Wordloss, summ_op = self.setup_performances_graph(self.ph_batch_T, 
                                                                self.ph_batch_mask, 
                                                                self.ph_batch_seqlen, 
                                                                batch_logits_flatted, 
                                                                batch_O_idx)
        
        train_op = self.setup_optimizer_graph(batch_Wordloss, self.ph_lr)
#         init_op = self.setup_initialization_graph()
        init_op = tf.global_variables_initializer()
        
        graph = tf.get_default_graph()
        return graph
    
    def setup_whole_graph_from_ckp(self, ckp_dir, ckp_prefix):
        '''
        import the graph saved in checkpoint as tf.default_graph()
        
        ckp_dir = ./ckp/{timestamp}/
        ckp_prefix = rnnlm_{str_acc}-{step}
        '''
        assert os.path.exists(ckp_dir), \
                'ckp_dir: %s not found' % (ckp_dir)
        
        print("Loading graph from : %s" % (ckp_dir + ckp_prefix))
        
        tf.reset_default_graph()
        saver = tf.train.import_meta_graph(ckp_dir + ckp_prefix + '.meta')
        return saver
    
    def is_Valid_Show_Gener_Save(self, 
                    step, 
                    valid_freq, 
                    show_freq,
                    gener_freq, 
                    save_freq
                    ):
        '''
        check the training step whether is:
            validation step,
            showing predictions step,
            generating sentences step,
            saving checkpoint step
            ...
        the step means #(trained instances)
        the *_freq means once every N batches   
        '''
        valid = step % (valid_freq * self.batch_size) == 0
        show = step % (show_freq * self.batch_size) == 0
        gener = step % (gener_freq * self.batch_size) == 0
        save = step % (save_freq * self.batch_size) == 0
        return valid, show, gener, save
    
    def validate_model(self, sess, valid_writer, step):
        '''
        validate the model in sess by validation set
        cross-entropy loss and accuracy are computed
        
        validation set: self.dp.valid_sentences
                        it is pushed into the model as one batch input
    
        PS: the sequences are truncated or padded into {seq_size} length
            so the results are approximative                        
        '''
        v_batchsize = len(self.dp.valid_sentences)
        v_batch_Hin = np.zeros([v_batchsize, self.hid_size * self.nb_layers])
        v_batcher = self.dp.minibatch_padding_masking_generater(
            self.dp.valid_sentences, 
            v_batchsize, 
            self.seq_size, 
            1)
        v_batch_X, v_batch_T, v_batch_Seqlen, v_batch_Mask, _ = next(v_batcher)
        
        v_feed_dict = {
            'ph_X:0':           v_batch_X,
            'ph_T:0':           v_batch_T,
            'ph_Hin:0':         v_batch_Hin,
            'ph_keep:0':        1.0,
            'ph_batchsize:0':   v_batchsize,
            'ph_Seqlen:0':      v_batch_Seqlen,
            'ph_Mask:0':        v_batch_Mask
        }
        
        v_loss, v_acc, v_summ = sess.run(['batch_loss:0', 'batch_acc:0', 'summ_op:0'],
                                         feed_dict = v_feed_dict)
        valid_writer.add_summary(v_summ, step)
        print("valid_loss : %.3f, valid_acc : %.1f%%" % (v_loss, v_acc * 100))
        return
    
    def show_model_predictions(self, sess, nb_shows):
        '''
        show {nb_shows} predictions from the model in sess
        the instances are randomly selected in the full dataset.
        the performances of the predictions, the comparisons between Target and Output
        are shown.
        '''
        s_batch_Hin = np.zeros([nb_shows, self.hid_size * self.nb_layers])
        idx_sampled, s_batch_X, s_batch_T, s_batch_Seqlen, s_batch_Mask = \
            self.dp.minibatch_sampled_padding_masking_generater(nb_shows, self.seq_size)
        
        s_feed_dict = {
            'ph_X:0':           s_batch_X,
            'ph_T:0':           s_batch_T,
            'ph_Hin:0':         s_batch_Hin,
            'ph_keep:0':        1.0,
            'ph_batchsize:0':   nb_shows,
            'ph_Seqlen:0':      s_batch_Seqlen,
            'ph_Mask:0':        s_batch_Mask
        }
        
        s_batch_O, s_loss, s_acc = sess.run(['O:0', 'batch_loss:0', 'batch_acc:0'],
                                            feed_dict = s_feed_dict)
        
        print("Show %d predictions in the full data" % (nb_shows))
        print("loss : %.3f, acc : %.1f%%" % (s_loss, s_acc * 100))
        
        for i in range(nb_shows):
            print("Sentence_id : %d, in Validation set : %s" % (
                idx_sampled[i],
                "No" if idx_sampled[i] < self.dp.idx_cut else "Yes")
                  )
            
            self.dp.print_aligned_comparison(
                s_batch_T[i], 
                s_batch_O[i], 
                max_length_to_show = 100, 
                true_len = s_batch_Seqlen[i])
        return
    
    def generate_sentences(self, sess, nb_gener):
        '''
        generate and show {nb_gener} sentences by the model in sess
        all sentences start by SentenceStartToken(SST)
                and are limited to seq_size
                or end by SentenceEndToken(SET)
        '''
        for i in range(nb_gener):
            print("Generated sentence %d" % (i))
            
            idx_o = self.dp.w2i[self.dp.start_token]
            print(self.dp.i2w[idx_o], end=' ')
            
            g_X = np.array([[idx_o]])
            g_Hin = np.zeros([1, self.hid_size * self.nb_layers])
            g_Seqlen = np.ones(g_X.shape[0])
            
            g_len = 0
            while idx_o != self.dp.w2i[self.dp.end_token] and g_len < self.seq_size:
                g_feed_dict = {
                    'ph_X:0':           g_X,
                    'ph_Hin:0':         g_Hin,
                    'ph_keep:0':        1.0,
                    'ph_batchsize:0':   1,
                    'ph_Seqlen:0':      g_Seqlen,
                }
                
                g_distr_o_flatted, g_Hin = sess.run(['distr_O_flatted:0', 'Ho:0'],
                                                    feed_dict = g_feed_dict)
                
                idx_o = self.dp.sample_from_probabilities(g_distr_o_flatted, topn = 3)
                print(self.dp.i2w[idx_o], end=' ')
                g_X = np.array([[idx_o]])
                g_len += 1
            print('\n')
        return
    
    def get_str_acc(self, acc, acc_points):
        '''
        grade the model according to its training accuracy:
        e.g. 
            acc_points = [0.2,0.5,0.9]
            acc < 0.2 -> str_acc = '20'
            acc < 0.5 -> str_acc = '50'
            acc < 0.9 -> str_acc = '90'
            acc >= 0.9 -> str_acc = 'gt90'
        '''
        str_acc = ''
        for i in range(len(acc_points)):
            if acc < acc_points[i]:
                str_acc = str(math.floor( acc_points[i] * 100 ))
                break
        if len(str_acc) == 0:
            str_acc = 'gt' + str(math.floor( acc_points[i] * 100 ))
        return str_acc
    
    def train_model(self,
                    nb_epochs,
                    valid_freq,
                    show_freq, 
                    gener_freq,
                    save_freq,
                    is_from_ckp = False,
                    ckp_config = None
                    ):
        '''
        train the model for {nb_epochs} epochs
        
        @is_from_ckp specifies whether training a new model 
                            or the model from checkpoint
        @ckp_config specifies the checkpoint model's directory and name prefix
                        {'dir':ckp_dir, 'prefix':ckp_prefix}
                        
        TensorFlow Summary Operations are logged under directory './log/{timestamp}/'
                    file 'training' for training summary statistic
                    file 'validation' for validation summary statistic
                    the whole graph structure is also saved in file 'training'
                    
        TensorFlow Checkpoints are saved under directory './ckp/{timestamp}/'
                    checkpoints name prefix pattern is 'rnnlm_{str_acc}-{step}'
                    only the current checkpoint will be saved in './ckp/{timestamp}/'
                    the last one will be remove into './ckp/{timestamp}/saved/' 
                        before the current one saved
        
        Besides the model's initial parameters and final parameters
                will be saved as a dictionary under directory './ckp/{timestamp}/'
                              named: init_dict and trained_dict
                for parameters checking in reload procession
                because of somehow no working in reloading                            
        '''
        timestamp = str(math.trunc(time.time()))
        '''
        preparation for TensorFlow Summary Writer
        '''
        train_writer = tf.summary.FileWriter("./log/" + timestamp + "/training")
        valid_writer = tf.summary.FileWriter("./log/" + timestamp + "/validation")
        '''
        show a progress bar that tells the progression in training {valid_freq} batches
        '''
        progress_bar = utils.Progress(valid_freq, size = 111+2,
                                      msg = "Training on next "+str(valid_freq)+" batches")
        
        save_dir = './ckp/' + timestamp + '/'
        '''
        preparation for checkpoint directory if not exists make it
        '''
        if not os.path.exists("./ckp/"):
            os.mkdir('./ckp/')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        
        '''setup the whole graph'''
        if is_from_ckp:
            old_dir = ckp_config['dir']
            old_prefix = ckp_config['prefix']
            saver = self.setup_whole_graph_from_ckp(old_dir, old_prefix)
        else:
            self.setup_whole_graph_from_codes()
            saver = tf.train.Saver(max_to_keep = 1)
                
        batch_Hin = np.zeros([self.batch_size, self.hid_size * self.nb_layers])
        
        '''start the Session, setup the graph and train the model'''
        with tf.Session() as sess:
            
            '''
            initialize all parameters
            setting uo the whole grah doesn't work here! 
            '''
            if is_from_ckp:
                old_dir = ckp_config['dir']
                old_prefix = ckp_config['prefix']
                old_path = old_dir + old_prefix
#                 saver = self.setup_whole_graph_from_ckp(old_dir, old_prefix)
                saver.restore(sess, old_path)
                print("continue training model from ckp %s" % (old_dir + old_prefix))
            else:
#                 self.setup_whole_graph_from_codes()
#                 saver = tf.train.Saver(max_to_keep = 1)
                sess.run('init')
                print('trian a new model at %s' % (save_dir))
            
            '''
            save the initial values of all parameters
            we can check whether the training has changed the values 
            '''
            init_dict = self.fetch_variable_value_dict(sess)
            with open(save_dir + 'init_dict', 'wb+') as file:
                pickle.dump(init_dict, file)
            
            '''add the graph structure into summaries'''
            train_writer.add_graph(sess.graph)
            
            step = 0
            tr_acc = 0.0
            '''
            start {nb_epochs} epochs training
            each batch of instances are generated form dp.*_generater
            '''
            for batch_X, batch_T, batch_Seqlen, batch_Mask, epoch in \
                self.dp.minibatch_padding_masking_generater(
                    self.dp.train_sentences, 
                    self.batch_size, 
                    self.seq_size, 
                    nb_epochs
                    ):
                '''check the step whether is a special step'''
                valid, show, gener, save = self.is_Valid_Show_Gener_Save(
                    step, 
                    valid_freq, 
                    show_freq, 
                    gener_freq, 
                    save_freq)
                
                '''organize the data feed into the model for training'''
                tr_feed_dict = {
                    'ph_X:0':           batch_X,
                    'ph_T:0':           batch_T,
                    'ph_Hin:0':         batch_Hin,
                    'ph_lr:0':          self.lr,
                    'ph_keep:0':        self.pr_keep,
                    'ph_batchsize:0':   self.batch_size,
                    'ph_Seqlen:0':      batch_Seqlen,
                    'ph_Mask:0':        batch_Mask
                }
                
                '''
                run one batch training
                summaries the loss and acc after training using this batch and dropout
                batch_Ho is the actual final hidden state
                        the hidden state for next batch may be it or just zero state
                        because word sentences start and end by SST and SET respectly
                '''
                _, batch_Ho, summ = sess.run(['trainer', 'Ho:0', 'summ_op:0'],
                                       feed_dict = tr_feed_dict)
                
                train_writer.add_summary(summ, global_step = step)
                
                if valid:
                    '''
                    organize the data feed into the model for validation
                        using the training batch without dropout
                    then validate the model by validation set and
                        summary the validation loss and acc
                    '''
                    valid_on_training_batch_feed_dict = {
                        'ph_X:0':           batch_X,
                        'ph_T:0':           batch_T,
                        'ph_Hin:0':         batch_Hin,
                        'ph_keep:0':        1.0,
                        'ph_batchsize:0':   self.batch_size,
                        'ph_Seqlen:0':      batch_Seqlen,
                        'ph_Mask:0':        batch_Mask
                    }
                    
                    tr_loss, tr_acc = sess.run(['batch_loss:0', 'batch_acc:0'],
                                               feed_dict = valid_on_training_batch_feed_dict)
                    
                    print("\ntrain over %4d step， (epoch %2d) :" % (step, epoch))
                    print("batch_loss : %.3f, batch_acc : %.1f%%" % (tr_loss, tr_acc * 100))
                    
                    self.validate_model(sess, valid_writer, step)
                
                '''
                show 10 predictions if model's training accuracy > 0.5
                it's wasty if acc is too low
                '''
                if show and tr_acc > 0.5:
                    self.show_model_predictions(sess, 10)
                
                '''show 5 generated sentences'''
                if gener and tr_acc > 0.5:
                    self.generate_sentences(sess, 5)
                    
                '''
                save the checkpoint
                if model's level is changed:
                    remove the last checkpoint to 'save_dir/saved/'
                else:
                    replace the last checkpoint
                '''
                if save:
                    acc_points = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
                    str_acc = self.get_str_acc(tr_acc, acc_points)
                    
                    file_prefix = save_dir + 'rnnlm_' + str_acc
                    
                    if not str_acc == self.get_str_acc(0.0, acc_points):
                        cp = utils.FileProcessor()
                        file_list = cp.find_files(file_prefix + '*')
                
                        if len(file_list) == 0:
                            last_files_pattern = save_dir + 'rnnlm_*'
                            files_to_transfer = cp.find_files(last_files_pattern)
                            cp.copy_files(files_to_transfer, save_dir + '/saved/')
                    
                    '''
                    save the trained values of all parameters
                    we can check whether reloading the values as the final trained values
                    save them instantly in case of termination before training finished
                    ''' 
                    trained_dict = self.fetch_variable_value_dict(sess)
                    with open(save_dir + 'trained_dict', 'wb+') as file:
                            pickle.dump(trained_dict, file)
                    saver.save(sess, file_prefix, global_step = step)
                    
                '''step to next batch'''
                progress_bar.step(reset = valid)
                
#                 batch_Hin = batch_Ho
                step += self.batch_size
        return
    
    def fetch_variable_value_dict(self, sess):
        '''
        fetch the parameters from the current session to a dictionary
        '''
        param_dict = {}
        for var in tf.trainable_variables():
            var_name = var.name
            var_value = sess.run(var)
            param_dict[var_name] = var_value
        return param_dict
    
    def test_trained_model(self, ckp_dir, ckp_prefix):
        '''
        test the trained model
        firstly, test whether the training changed the parameter values
        secondly, test whether reloading parameter values correctly
        lastly, test whether the model perform normally
        '''
        print("Loading trained model from : %s" % ckp_dir + ckp_prefix)
        saver = tf.train.import_meta_graph(ckp_dir + ckp_prefix + '.meta')
        
        with open(ckp_dir + 'init_dict', 'rb') as file:
            init_dict = pickle.load(file)
        with open(ckp_dir + 'trained_dict', 'rb') as file:
            trained_dict = pickle.load(file)
        
        with tf.Session() as sess:
            
            saver.restore(sess, ckp_dir + ckp_prefix)
            
            cur_dict = self.fetch_variable_value_dict(sess)
            
            print('dict ckeck:')
            is_cur_equal_init = utils.dict_comparison(cur_dict, init_dict)
            is_cur_equal_trained = utils.dict_comparison(cur_dict, trained_dict)
            is_init_equal_trained = utils.dict_comparison(init_dict, trained_dict)
            print('cur == init : ' + str(is_cur_equal_init))
            print('cur == trained : ' + str(is_cur_equal_trained))
            print('init == trained : ' + str(is_init_equal_trained))
            
            self.show_model_predictions(sess, 10)
            self.generate_sentences(sess, 10)
            
        return


if __name__ == '__main__':
    
    SEQLEN = 48
    BATCHSIZE = 32
    VOCABSIZE = 3000
    EMBEDSIZE = 128
    INTERNALSIZE = 100
    NLAYERS = 2
    learning_rate = 0.001
    dropout_pkeep = 0.8    # 1.0 => no dropout
    
    datadir = 'E:/data/phd/datasets/childrenreading/the_little_prince.csv'
    
    ''' first run, save DataProcessor, train model, save checkpoint '''
#     dp = utils.DataProcessor()
#     dp.loadLittlePrince(datadir, VOCABSIZE)
#     dp.save_data_processor()
#     lm = rnnlm(data_processor = dp, 
#                batch_size = BATCHSIZE,
#                seq_size = SEQLEN,
#                hid_size = INTERNALSIZE,
#                nb_layers = NLAYERS,
#                embed_size = EMBEDSIZE,
#                learning_rate = learning_rate,
#                pr_keep = dropout_pkeep
#                )
#     lm.train_model(
#         nb_epochs = 1000, 
#         valid_freq = 20, 
#         show_freq = 40, 
#         gener_freq = 60, 
#         save_freq = 80, 
#         is_from_ckp = False, 
#         ckp_config = None
#         )
    
    ''' later run load DataProcessor and checkpoint, test/continue train the model'''
    dp = utils.DataProcessor()
    dp.load_data_processor()
    lm = rnnlm(data_processor = dp, 
               batch_size = BATCHSIZE,
               seq_size = SEQLEN,
               hid_size = INTERNALSIZE,
               nb_layers = NLAYERS,
               embed_size = EMBEDSIZE,
               learning_rate = 0.001,
               pr_keep = 0.75
               )
     
    ckp_config = {'dir':'./ckp/1491976098/', 'prefix':'rnnlm_90-693760'}
     
#     lm.test_trained_model(ckp_config['dir'], ckp_config['prefix'])
     
    lm.train_model(
        nb_epochs = 1000, 
        valid_freq = 20, 
        show_freq = 40, 
        gener_freq = 60, 
        save_freq = 80,
        is_from_ckp = True, 
        ckp_config = ckp_config
        )
    
    
    
        
    
    