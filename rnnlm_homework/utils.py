'''
Created on 2017年3月29日

@author: Aevil
'''
import os
import shutil
import glob
import pickle
import itertools
import nltk
import numpy as np
import matplotlib.pyplot as plt


class FileProcessor(object):
    
    def find_files(self, file_pattern):
        '''find the files matched with {file_pattern}'''
        file_list = glob.glob(file_pattern, recursive = True)
        if len(file_list) == 0:
            print("No file found like %s" % (file_pattern))
        return file_list
    
    def copy_files(self, file_list, dest_dir):
        '''
        copy the files to the directory {dest_dir}
        @file_list is a list of paths of these files
        '''
        if len(file_list) == 0:
            print("No file copied")
            return
        if not os.path.exists(dest_dir):
            os.mkdir(dest_dir)
        for file in file_list:
            shutil.copy(file, dest_dir)
        return

class DataProcessor(object):

    def __init__(self):
        '''define SentenceStartToken, SentenceEndToken and UNKnownWordToken'''
        self.start_token = 'SST'
        self.end_token = 'SET'
        self.unk_token = 'UNK'
    
    def loadLittlePrince(self, path, vocabulary_size = 3000, show_distr = False):
        '''
        load the 'The little Prince' text 
        create the vocabulary table from the text
        map: word <-> index in vocabulary
        split the whole set of sentences into training set and validation set
        
        PS: this function use the famous Natural Language Toolkit 
            to tokenize the sentences and word
            to create vocabulary table
            so before run it, make sure nltk package, english pickle are setup 
        '''
        sentence_start_token = self.start_token
        sentence_end_token = self.end_token
        unknown_token = self.unk_token
        print ("Reading The Little Prince...")
        with open(
            path, 
            'rt',
            encoding='utf-8'
            ) as f:
            
            '''
            read each paragraph
            remove the '\u3000', '"' from the paragraph
            tokenize the paragraph into sentences
            add SST and SET to the head and tail of each sentence 
            '''
            sentences = []
            for line in f:
                paragraph_i = nltk.sent_tokenize(line.replace('\u3000', '').replace("\"","").lower())
                sentences += paragraph_i                
            sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
        
        '''tokenize each sentence to words'''
        tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
        '''statistic the frequency distribution of words'''
        word_freq = nltk.FreqDist(itertools.chain.from_iterable(tokenized_sentences))
        print ("Found %d unique words tokens." % len(word_freq.items()))
        
        '''
        create vocabulary table for top {vocabulary_size-1} words in FreqDistr
        remain 1 position for UNK
        if #words < {vocabulary_size-1}:
            using all words
        '''
        vocab = word_freq.most_common(vocabulary_size-1)
        '''create mapping'''
        index_to_word = [x[0] for x in vocab]
        index_to_word.append(unknown_token)
        word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
        
        print ("Using vocabulary size %d." % len(index_to_word))
        print ("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))
        
        '''
        replace the word -> UNK if it not in vocabulary table
        maping the word -> index for each sentence 
        '''
        tokenized_sentences = [ [w if w in word_to_index else unknown_token for w in sent] for sent in tokenized_sentences ]
        idx_sentences = [ [word_to_index[w] for w in sent] for sent in tokenized_sentences]
        print ("\nExample sentence: '%s'" % sentences[-1])
        print ("\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[-1])
        
        '''
        statistic sentence length information for decision of seq_size in rnnlm
        '''
        len_sentences = np.array([len(sent) for sent in tokenized_sentences])
        len_sentences.sort()
                
        nb_sentences = len(sentences)
        min_sent_len = int( np.min(len_sentences) )
        max_sent_len = int( np.max(len_sentences) )
        median_sent_len = int( np.median(len_sentences) )
        mean_sent_len = int( np.mean(len_sentences) )
        
        
        print ("\nParsed %d sentences" % (nb_sentences))
        print ("Min length of sentences : %d" % (min_sent_len))
        print ("Max length of sentences : %d" % (max_sent_len))
        print ("Median length of sentences : %d" % (median_sent_len))
        print ("Mean length of sentences : %d\n" % (mean_sent_len))
        
        if show_distr:
            percents_list = np.array([0.5, 0.7, 0.9, 0.95, 0.99, 1-(1e-5)])
            bounds = self.plot_len_distribution(len_sentences, percents_list)
             
            for i in range(len(bounds)):
                print("%.0f%% sentences under the length of %d" % (percents_list[i] * 100, 
                                                                   bounds[i]))
        
        '''keep data needed'''
        self.idx_sentences = idx_sentences
        self.vocab = vocab
        self.i2w = index_to_word
        self.w2i = word_to_index
        
        self.vocabsize = len(index_to_word)
        self.nb_sents = len(idx_sentences)
        self.maxlen = max_sent_len
        self.medianlen = median_sent_len
        self.meanlen = mean_sent_len
#         print(idx_sentences)
#         print(vocab)
        train_rate = 0.7
        self.idx_cut = int( self.nb_sents * train_rate )
        print("\nTraining data (%.1f%%): No. 1 ~ %d sentences" % (
            train_rate*100, 
            self.idx_cut)
              )
        print("Validation data (%.1f%%): No. %d ~ %d sentences" % (
            (1 - train_rate) * 100, 
            self.idx_cut + 1, 
            self.nb_sents)
              )
        self.train_sentences = self.idx_sentences[:self.idx_cut]
        self.valid_sentences = self.idx_sentences[self.idx_cut:]
        
        print ("\nThe Little Prince loaded successfully")
        return self.train_sentences, self.valid_sentences
    
    def load_pretrained_embeddings(self, embed_dir, embed_size=100):        
        assert os.path.exists(embed_dir), "embed_dir does not exist"
        assert self.w2i is not None, "vocabulary does not exist"
        
        embeddings = np.random.uniform(
            low = -1.0, 
            high = 1.0, 
            size = (self.vocabsize, embed_size)
            )
        nb_pretrained = 0
        with open(
            embed_dir + 'glove.6B.%dd.txt' % (embed_size), 
            'rt',
            encoding='utf-8'
            ) as f:
            for line in f:
                line_split = line.split()
                w = line_split[0]
                if w in self.w2i.keys():
                    w_idx = self.w2i[w]
                    w_embedding = np.array(line_split[1:])
                    w_embedding = w_embedding.astype(np.float32)
                    embeddings[w_idx] = w_embedding
                    nb_pretrained += 1
        
        self.embeddings = embeddings
        print("load pretrained embeddings %s success" % (str(self.embeddings.shape)))
        print("#pretrained/#vocabulary = %.1f%%(%d/%d)" % (
            nb_pretrained * 100.0 / self.vocabsize,
            nb_pretrained,
            self.vocabsize
            ))
        return self.embeddings
    
    def words_between_train_and_valid(self, training, validation, i2w=None):
        train_words = set()
        for sent in training:
            train_words |= set(sent)
        valid_words = set()
        for sent in validation:
            valid_words |= set(sent) 
        
        all_words = train_words | valid_words        
        inter = train_words & valid_words
        
        tr_rate = len(train_words) * 100.0 / len(all_words)
        va_rate = len(valid_words) * 100.0 / len(all_words)
        com_v_tr = len(inter) * 100.0 / len(train_words)
        com_v_va = len(inter) * 100.0 / len(valid_words)
        
        print("training set (%.1f%%) and validation set (%.1f%%) have %d words in common" % (
            tr_rate,
            va_rate,
            len(inter)
            ))
        print("%.1f%% in training set, %.1f%% in validation set" % (com_v_tr, com_v_va))
        
        if i2w is not None:
            inter = list(inter)
            inter.sort(reverse=False)
            inter_word = [self.i2w[idx] for idx in inter]        
            print("they are :")        
            print(inter_word)
        return
    
    def plot_len_distribution(self, lens_sorted, percents_list):
        nb_sentences = len(lens_sorted)
        idx_list = np.int32(np.floor( percents_list * nb_sentences))
        bound_list = lens_sorted[idx_list]
        plt.hist(lens_sorted, bins=100)
        locs, _ = plt.yticks()        
        plt.vlines(bound_list, ymin=locs[0], ymax=locs[-1], colors='red')
        for i  in range(len(percents_list)):
            plt.text(
                bound_list[i], 
                locs[-1] * percents_list[i], 
                s = "%.0f%%" % (percents_list[i] * 100),
                color = 'orange',
                va = 'top'
                )
            plt.text(
                bound_list[i], 
                locs[0]-5, 
                s = "%d" % (bound_list[i]),
                color = 'red',
                ha = 'center',
                va = 'top'
                )
            
        plt.title("sentence length distribution")
        plt.ylabel('count')
        plt.xlabel('sentence length')
        plt.show()
        return bound_list
        
        
    
    def get_sequence_from_sentences(self, sentences):
        return list(itertools.chain.from_iterable(sentences))
    
    def minibatch_nopadding_generater(self, idx_sequence, batch_size, sequence_size, nb_epochs):
        data = np.array(idx_sequence)
        data_len = data.shape[0]
        # using (data_len-1) because we must provide for the sequence shifted by 1 too
        nb_batches = (data_len - 1) // (batch_size * sequence_size)
        assert nb_batches > 0, "Not enough data, even for a single batch. Try using a smaller batch_size."
        rounded_data_len = nb_batches * batch_size * sequence_size
        xdata = np.reshape(data[0:rounded_data_len], [batch_size, nb_batches * sequence_size])
        ydata = np.reshape(data[1:rounded_data_len + 1], [batch_size, nb_batches * sequence_size])

        for epoch in range(nb_epochs):
            for batch in range(nb_batches):
                x = xdata[:, batch * sequence_size:(batch + 1) * sequence_size]
                y = ydata[:, batch * sequence_size:(batch + 1) * sequence_size]
                x = np.roll(x, -epoch, axis=0)  # to continue the text from epoch to epoch (do not reset rnn state!)
                y = np.roll(y, -epoch, axis=0)
                yield x, y, epoch
    
    def minibatch_padding_masking_generater(self, idx_sentences, batch_size, sequence_size, nb_epochs):
        data = np.array(idx_sentences)
        nb_sentences = data.shape[0]
        nb_batches = nb_sentences // batch_size
        assert nb_batches > 0, "Not enough data, even for a single batch. Try using a smaller batch_size."
        rounded_data_len = nb_batches * batch_size
        data_trunctated = np.reshape(data[:rounded_data_len],[batch_size, nb_batches])
        for epoch in range(nb_epochs):
            for batch in range(nb_batches):
                data_b = data_trunctated[:, batch]
                x = np.array([np.resize(x_i[:-1], (sequence_size,)) for x_i in data_b])
                y = np.array([np.resize(x_i[1:], (sequence_size,)) for x_i in data_b])
                slens = np.array([min(sequence_size, len(x_i)-1) for x_i in data_b])
                mask = np.array([ [1]*min(sequence_size, slens[r]) + [0]*max(0, sequence_size - slens[r]) for r in range(batch_size) ])
                
                x = np.roll(x, -epoch, axis=0)
                y = np.roll(y, -epoch, axis=0)
                slens = np.roll(slens, -epoch, axis=0)
                mask = np.roll(mask, -epoch, axis=0)
                yield x, y, slens, mask, epoch
    
    def nobatch_generater(self, idx_sentences, nb_epochs):
        for epoch in range(nb_epochs):
            for idx_sent in idx_sentences:
                x = np.array([idx_sent[:-1]])
                y = np.array([idx_sent[1:]])
                yield x, y, epoch
    
    def minibatch_dynamic_padding_masking_generater(self, idx_sentences, batch_size, nb_epochs):
        data = np.array(idx_sentences)
        nb_sentences = data.shape[0]
        nb_batches = nb_sentences // batch_size
        assert nb_batches > 0, "Not enough data, even for a single batch. Try using a smaller batch_size."
        rounded_data_len = nb_batches * batch_size
        data_trunctated = np.reshape(data[:rounded_data_len],[batch_size, nb_batches])
        for epoch in range(nb_epochs):
            for batch in range(nb_batches):
                data_b = data_trunctated[:, batch]
                slens = np.array([len(x_i)-1 for x_i in data_b])
                sequence_size = np.max(slens)
                x = np.array([np.resize(x_i[:-1], (sequence_size,)) for x_i in data_b])
                y = np.array([np.resize(x_i[1:], (sequence_size,)) for x_i in data_b])                
                mask = np.array([ [1]*min(sequence_size, slens[r]) + [0]*max(0, sequence_size - slens[r]) for r in range(batch_size) ])
                
                x = np.roll(x, -epoch, axis=0)
                y = np.roll(y, -epoch, axis=0)
                slens = np.roll(slens, -epoch, axis=0)
                mask = np.roll(mask, -epoch, axis=0)
                yield x, y, slens, mask, epoch
    
    def minibatch_sampled_padding_masking_generater(self, batch_size, sequence_size):
        idx_sampled = np.random.choice(self.nb_sents, batch_size)
        idx_sampled_sents = np.asarray(self.idx_sentences)[idx_sampled]
        x = np.array([np.resize(x_i[:-1], (sequence_size,)) for x_i in idx_sampled_sents])
        y = np.array([np.resize(x_i[1:], (sequence_size,)) for x_i in idx_sampled_sents])
        slens = np.array([min(sequence_size, len(x_i)-1) for x_i in idx_sampled_sents])
        mask = np.array([ [1]*min(sequence_size, slens[r]) + [0]*max(0, sequence_size - slens[r]) for r in range(batch_size) ])
        return idx_sampled, x, y, slens, mask
    
    def sample_from_probabilities(self, prob_distr, topn = 1):
        p = np.squeeze(prob_distr)
        p[np.argsort(p)[:-topn]] = 0
        p = p / np.sum(p)
        return np.random.choice(self.vocabsize, 1, p=p)[0]
    
    def print_aligned_comparison(self, 
                                  idx_sentence_t, 
                                  idx_sentence_o, 
                                  max_length_to_show,
                                  true_len = None
                                  ):
        show_len = 0
        show_strs = []
        str_t = "|"
        str_o = "|"
        
        max_idx = max(len(idx_sentence_t), len(idx_sentence_o)) - 1        
        if true_len is not None:
            max_idx = true_len - 1
        
        for i in range(max_idx + 1):
            w_t = ""
            w_o = ""
            if i < min(len(idx_sentence_t), len(idx_sentence_o)):
                w_t = self.i2w[idx_sentence_t[i]]
                w_o = self.i2w[idx_sentence_o[i]]
            elif i < len(idx_sentence_t):
                w_t = self.i2w[idx_sentence_t[i]]
            else:
                w_o = self.i2w[idx_sentence_o[i]]
            
            w_max_len = max(len(w_t), len(w_o))
            w_string_format = "%" +str(w_max_len)+"s"
                
            if show_len + w_max_len +1 <= max_length_to_show:
                w_string_format += ' '
                str_t += w_string_format%(w_t)
                str_o += w_string_format%(w_o)
                show_len += w_max_len +1
            elif show_len + w_max_len <= max_length_to_show:
                str_t += w_string_format%(w_t)
                str_o += w_string_format%(w_o)
                show_len += w_max_len
            else:
                str_t += '\n' + str_o
                show_strs.append(str_t)
                
                w_string_format += ' '
                str_t = w_string_format%(w_t)
                str_o = w_string_format%(w_o)
                show_len = w_max_len + 1
            
            if i == max_idx:
                str_t += '|\n' + str_o + '|\n'
                show_strs.append(str_t)
        
        for doubleline in show_strs:
            print(doubleline)
        return
                
    def save_data_processor(self, save_dir = './dp/'):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        
        var_dict = vars(self)        
        with open(save_dir + 'data_processor', 'wb+') as file:
            pickle.dump(var_dict, file)
        
        print("data processor saved under %s" % (save_dir))
        return
    
    def load_data_processor(self, load_dir = './dp/'):
        assert os.path.exists(load_dir), 'Directory %s not found' % (load_dir)
        with open(load_dir + 'data_processor', 'rb') as file:
            var_dict = pickle.load(file)
        for name in var_dict.keys():
            vars(self)[name] = var_dict[name]
        print("data processor loaded from %s" % (load_dir))
        return
    
    def test_display_all_members(self):
        for name,value in  vars(self).items():
            print(name)
            print(value)
        return
    
    def test_comparison(self):
        nb_test = 10
        for i in range(nb_test -1 ):
            choice = np.random.choice(self.nb_sents, 2)
            print("test %d, sent_t %d, sent_o %d" % (i, choice[0], choice[1]))
            self.print_aligned_comparison(
                self.idx_sentences[choice[0]], 
                self.idx_sentences[choice[1]], 
                max_length_to_show = 100)
        
        choice = np.random.choice(self.nb_sents, 1)
        print("test %d, sent_t %d, sent_o %d" % (i, choice[0], choice[0]))
        self.print_aligned_comparision(
            self.idx_sentences[choice[0]], 
            self.idx_sentences[choice[0]], 
            max_length_to_show = 100)        
        return
    
    def test_sampled_minibatch(self):
        batchsize = 10
        seqsize = 20
        idx_sampled, bx, _, bsl, _ = self.minibatch_sampled_padding_masking_generater(
            batchsize, seqsize)
        for i in range(batchsize):
            print("select_id %d, ori_len %d, sampled_len %d, true_len %d" % (
                    idx_sampled[i], 
                    len(self.idx_sentences[idx_sampled[i]]) - 1,
                    len(bx[i]),
                    bsl[i])
                  )
            self.print_aligned_comparison(self.idx_sentences[idx_sampled[i]], 
                                           bx[i], 
                                           max_length_to_show = 100, 
                                           true_len = bsl[i]
                                           )
        return
    
class Progress:
    """Text mode progress bar.
    Usage:
            p = Progress(30)
            p.step()
            p.step()
            p.step(start=True) # to restart form 0%
    The progress bar displays a new header at each restart."""
    def __init__(self, maxi, size=100, msg=""):
        """
        :param maxi: the number of steps required to reach 100%
        :param size: the number of characters taken on the screen by the progress bar
        :param msg: the message displayed in the header of the progress bat
        """
        self.maxi = maxi
        self.p = self.__start_progress(maxi)()  # () to get the iterator from the generator
        self.header_printed = False
        self.msg = msg
        self.size = size

    def step(self, reset=False):
        if reset:
            self.__init__(self.maxi, self.size, self.msg)
        if not self.header_printed:
            self.__print_header()
        next(self.p)

    def __print_header(self):
        print()
        format_string = "0%{: ^" + str(self.size - 6) + "}100%"
        print(format_string.format(self.msg))
        self.header_printed = True

    def __start_progress(self, maxi):
        def print_progress():
            # Bresenham's algorithm. Yields the number of dots printed.
            # This will always print 100 dots in max invocations.
            dx = maxi
            dy = self.size
            d = dy - dx
            for x in range(maxi):
                k = 0
                while d >= 0:
                    print('=', end="", flush=True)
                    k += 1
                    d -= dx
                d += dy
                yield k

        return print_progress
    
def dict_comparison(dict1, dict2, is_show_differences = False):
    '''
    compare the dictionaries
    if all key:value are equal between two dicts return True
    else False
    '''
    equal_flag = True
    if not dict1.keys() == dict2.keys():
        equal_flag = False
    else:
        for key in dict1.keys():
            value_cmp = np.asarray(dict1[key] == dict2[key]).all()
            if value_cmp == False:
                if equal_flag == True:
                    equal_flag =False
                if is_show_differences:
                    print('Differences @ %s' % (key))
                    print(dict1[key] - dict2[key])
    return equal_flag

if __name__ == '__main__':
    
    tlp_dir = 'E:/data/phd/datasets/childrenreading/the_little_prince.csv'
    imdb_dir = 'E:/data/phd/datasets/imdb/imdb.npz'
    embed_dir = 'E:/data/phd/datasets/glove.6B/'
    dp1 = DataProcessor()
    dp2 = DataProcessor()
        
#     dp1.loadLittlePrince(tlp_dir)
#     dp1.load_pretrained_embeddings(embed_dir)
#     dp1.words_between_train_and_valid(dp1.train_sentences, dp1.valid_sentences)
#     dp1.save_data_processor()
    
#     dp2.load_data_processor()
#     dp1.test_display_all_members()
#     dp2.test_display_all_members()
    
#     print(dict_comparison(dp1.w2i, dp2.w2i))
#     dp2.test_vocab_i2w_w2i_order()


    