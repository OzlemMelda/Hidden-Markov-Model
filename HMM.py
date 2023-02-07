import numpy as np

import argparse
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import itertools
import pickle


class SingleHMM:
    """
    Hidden Markov Model for single observation
    """
    def __init__(self, n_states=10, vocab_size=256):
        """
        initialize parameters (pi, transitions, emissions)
        """
        self.n_states = n_states
        self.vocab_size = vocab_size

        self.prior_knowledge = np.random.rand(self.n_states)
        self.prior_knowledge /= self.prior_knowledge.sum()

        self.transition_mat = np.random.uniform(
            size=(self.n_states, self.n_states))
        self.transition_mat /= self.transition_mat.sum(
            axis=1, keepdims=True)

        self.emission_mat = np.random.uniform(
            size=(self.n_states, self.vocab_size))
        self.emission_mat /= self.emission_mat.sum(
            axis=1, keepdims=True)

        self.alpha = None
        self.beta = None
        self.c_arr = None
        self.gamma = None
        self.xi = None

    def forward(self, sequence):
        """
        compute alpha / alpha-pass
        """
        self.alpha = np.zeros((len(sequence), self.n_states))
        self.c_arr = []

        c0 = 0
        for i in range(self.n_states):
            self.alpha[0, i] = self.prior_knowledge[i] * \
                               self.emission_mat[i, sequence[0]]
            c0 += self.alpha[0, i]
        c0 = 1/c0
        self.c_arr.append(c0)

        # scale alpha zero
        self.alpha[0] = c0 * self.alpha[0]

        for t in range(1, len(sequence)):
            ct = 0
            for i in range(self.n_states):
                self.alpha[t, i] = 0
                for j in range(self.n_states):
                    self.alpha[t, i] += self.alpha[t-1, j] * \
                                       self.transition_mat[j, i]
                self.alpha[t, i] = self.alpha[t, i] * \
                                   self.emission_mat[i, sequence[t]]
                ct += self.alpha[t, i]
            ct = 1/ct
            self.c_arr.append(ct)
            self.alpha[t] *= ct

    def backward(self, sequence):
        """
        compute beta / beta-pass
        """
        T = len(sequence)
        self.beta = np.zeros((T, self.n_states))
        self.beta[T-1] = np.full((self.n_states,), self.c_arr[T-1],
                                 dtype=np.float)

        for t in range(T-2, -1, -1):
            for i in range(self.n_states):
                self.beta[t, i] = 0
                for j in range(self.n_states):
                    self.beta[t, i] += self.transition_mat[i, j] * \
                                       self.emission_mat[j, sequence[t]] * \
                                       self.beta[t+1, j]
                self.beta[t, i] *= self.c_arr[t]

    def get_gamma_xi(self, sequence):
        T = len(sequence)
        self.xi = np.zeros((T-1, self.n_states, self.n_states))
        self.gamma = np.zeros((T, self.n_states))
        
        for t in range(T-1):
            for i in range(self.n_states):
                self.gamma[t, i] = 0 
                for j in range(self.n_states):
                    self.xi[t, i, j] = self.alpha[t, i] * \
                                       self.transition_mat[i, j] * \
                                       self.emission_mat[j, sequence[t+1]] * \
                                       self.beta[t+1, j]
                    self.gamma[t, i] += self.xi[t, i, j]
        
        self.gamma[T-1] = self.alpha[T-1].copy()

    def e_step(self, sequence):
        """
        compute alpha, beta, gamma, xi given
        transition / emission / pi
        """
        self.forward(sequence)
        self.backward(sequence)
        self.get_gamma_xi(sequence)

    def m_step(self, sequence):
        """
        re-estimate transition / emission matrix / pi
        given alpha, beta, gamma, xi
        """
        T = len(sequence)

        # new estimate for prior knowledge
        self.prior_knowledge = self.gamma[0].copy()

        # new estimate for transition matrix
        for i in range(self.n_states):
            den = 0
            for t in range(T-1):
                den += self.gamma[t, i]
            for j in range(self.n_states):
                num = 0
                for t in range(T-1):
                    num += self.xi[t, i, j]
                self.transition_mat[i, j] = num/den
        
        # new estimate for emission matrix 
        for i in range(self.n_states):
            den = 0
            for t in range(T):
                den += self.gamma[t, i]
            
            for o in range(self.vocab_size):
                num = 0
                for t in range(T):
                    if sequence[t] == o:
                        num += self.gamma[t, i]
            self.emission_mat[i, o] = num/den

        # row-wise normalization of emission matrix
        self.emission_mat /= self.emission_mat.sum(axis=1,
                                                   keepdims=True)

    def get_evidence(self):
        log_prob = 0.
        for ct in self.c_arr:
            log_prob += np.log(ct)
        return -log_prob


class MultiHMM:
    """
    Hidden Markov Model for multiple observations
    """

    def __init__(self,
                 n_states,
                 vocab_size):
        self.hmm = SingleHMM(n_states, vocab_size)
        self.alpha_list = []
        self.gamma_list = []
        self.xi_list = []
        self.evidence = 0.

        # own parameters
        self.prior_knowledge = None
        self.transition_mat = None
        self.emission_mat = None

    def e_step(self, multiple_sequence):
        self.evidence = 0.
        self.alpha_list.clear()
        self.gamma_list.clear()
        self.xi_list.clear()
        for seq in multiple_sequence:
            self.hmm.e_step(seq)
            self.alpha_list.append(self.hmm.alpha)
            self.gamma_list.append(self.hmm.gamma)
            self.xi_list.append(self.hmm.xi)
            self.evidence += self.hmm.get_evidence()
        self.evidence = self.evidence / len(multiple_sequence)

    def test(self, sequence):
        self.hmm.e_step(sequence)
        self.alpha_list.append(self.hmm.alpha)
        self.gamma_list.append(self.hmm.gamma)
        self.xi_list.append(self.hmm.xi)

        return self.hmm.get_evidence()

    def m_step(self, multiple_sequence): 
        num_seq = len(multiple_sequence)
        self.prior_knowledge = np.zeros((self.hmm.n_states,),
                                        dtype=np.float)
        self.emission_mat = np.zeros((self.hmm.n_states,
                                      self.hmm.vocab_size),
                                     dtype=np.float)

        num_t = np.zeros((self.hmm.n_states, self.hmm.n_states),
                           dtype=np.float)
        den_t = np.zeros((self.hmm.n_states, 1),
                           dtype=np.float)
        num_e = np.zeros((self.hmm.n_states, self.hmm.vocab_size),
                           dtype=np.float)
        den_e = np.zeros((self.hmm.n_states, 1),
                           dtype=np.float)

        for r in range(num_seq):
            self.prior_knowledge += self.gamma_list[r][0]
            for t, o in enumerate(multiple_sequence[r]):
                num_e[:, o] += self.gamma_list[r][t]
                den_e[:, 0] += self.gamma_list[r][t]
                if t < len(multiple_sequence[r])-1:
                    num_t += self.xi_list[r][t]
                    den_t[:, 0] += self.gamma_list[r][t]
                
        self.prior_knowledge /= num_seq
        self.transition_mat = num_t / den_t
        self.emission_mat = num_e / den_e

        # update params for single hmm
        self.hmm.prior_knowledge = self.prior_knowledge.copy()
        self.hmm.transition_mat = self.transition_mat.copy()
        self.hmm.emission_mat = self.emission_mat.copy()

    def get_evidence(self):
        return self.evidence


class Convergence:
    """
    Check if converge
    """
    def __init__(self):
        self.results = [-np.inf]

    def update(self, new_result):        
        stop_update = True if new_result < self.results[-1] else False
        self.results.append(new_result)
        return stop_update 


def load_subdir(path):
    """
    load data
    """
    data = []
    for filename in os.listdir(path):
        with open(os.path.join(path, filename)) as fh:
            data.append(fh.read())
    return data


def clean_sentences(df):
    """
    clean english words from sentences such as I, you, we, they, would, can, may, ...
    """
    tokenizer = RegexpTokenizer(r'\w+')
    nltk_tokens = [tokenizer.tokenize(rr) for rr in df]
    stopwords_set = set(stopwords.words('english'))
    training_list = [list(set(sentence).difference(stopwords_set)) for sentence in nltk_tokens]

    return training_list


def create_dictionary(df, word_dict_size):
    """
    create dictionary using most frequent words
    """
    training_word_list = list(itertools.chain.from_iterable(df))
    training_freq_list = nltk.FreqDist(training_word_list)
    del training_word_list

    training_most_common = []
    for word, frequency in training_freq_list.most_common(word_dict_size - 1):
        training_most_common.append(word)
    training_most_common.append('UNK')
    training_dictionary = dict([(y, x) for x, y in enumerate(sorted(set(training_most_common)))])
    del training_most_common

    return training_dictionary


def assign_index(df, dictionary):
    """
    assign index to each word using dictionary.
    if a word is not in dictionary, assign UNK index to the corresponding word
    """
    training_list_id = []
    unk_id = dictionary['UNK']
    for sent in df:
        sent_id = []
        for word in sent:
            word_id = dictionary.get(word, unk_id)
            sent_id.append(word_id)
        training_list_id.append(sent_id)

    return training_list_id


def save_model(model, save_path):
    """ save """
    with open(save_path, 'wb') as outp:
        pickle.dump(model, outp, pickle.HIGHEST_PROTOCOL)


def load_model(load_path):
    """ load """
    with open(load_path, 'rb') as inp:
        model = pickle.load(inp)
    return model


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Program to build and train a neural network.')
    parser.add_argument('--dev_path', default=None, help='Path to development (i.e., testing) data.')
    parser.add_argument('--train_path', default=None, help='Path to the training data directory.')
    parser.add_argument('--max_iters', type=int, default=30, help='The maximum number of EM iterations (default 30)')
    parser.add_argument('--model_out', default=None, help='Filename to save the final model.')
    parser.add_argument('--dict_out', default=None, help='File name to save dictionary.')
    parser.add_argument('--hidden_states', type=int, default=10, help='The number of hidden states to use. (default 10)')
    parser.add_argument('--row_count', type=int, default=2500, help='The number of training data to use. (default 2500)')
    parser.add_argument('--vocab_size', type=int, default=256,
                        help='The number of frequent words in the dictionary, others are Unknown Words. (default 256)')

    args = parser.parse_args()

    # data preprocessing
    # clean sentences from English words
    # create frequent word list and assign index to each frequent word
    # convert words to indexes in input data
    training_data = load_subdir(os.path.join(args.dev_path, args.train_path))
    training_data = clean_sentences(training_data)
    dictionary = create_dictionary(training_data, args.vocab_size)
    training_data = assign_index(training_data, dictionary)

    # select size of training data
    obs = training_data[0:args.row_count]
    del training_data

    # construct
    model = MultiHMM(args.hidden_states, args.vocab_size)
    monitor = Convergence()

    # train
    for it in range(args.max_iters):
        before = model.get_evidence()
        model.e_step(obs)
        model.m_step(obs)
        print('ITER {:02d} | BEFORE loglike : {:06f} AFTER loglike : {:06f}'.format(it+1, before, model.get_evidence()))
        if monitor.update(model.get_evidence()):
            break

    print(f"transition matrix : {model.transition_mat}")
    print(f"emission matrix : {model.emission_mat}")
    print(f"initial hidden states :{model.prior_knowledge}")

    # save model and dictionary
    model.xi_list = []
    model.alpha_list = []
    model.gamma_list = []

    save_model(model,
               args.model_out)

    # save dictionary
    save_model(dictionary,
               args.dict_out)

