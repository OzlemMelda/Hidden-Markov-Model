# File: classify.py
# Purpose:  Starter code for the main experiment for CSC 246 P3 F22.

import argparse
from HMM import *


def main():
    parser = argparse.ArgumentParser(description='Program to build and train a neural network.')
    parser.add_argument('--pos_hmm', default=None, help='Path to the positive class hmm.')
    parser.add_argument('--neg_hmm', default=None, help='Path to the negative class hmm.')
    parser.add_argument('--datapath', default=None, help='Path to the test data.')
    parser.add_argument('--pos_dictpath', default=None, help='Path to the positive hmm dictionary.')
    parser.add_argument('--neg_dictpath', default=None, help='Path to the negative hmm dictionary.')

    args = parser.parse_args()

    # Load HMMs
    pos_hmm = load_model(args.pos_hmm)
    neg_hmm = load_model(args.neg_hmm)
    pos_dictionary = load_model(args.pos_dictpath)
    neg_dictionary = load_model(args.neg_dictpath)

    correct = 0
    total = 0

    # clean and test samples from positive data path
    samples = load_subdir(os.path.join(args.datapath, 'pos'))
    samples = clean_sentences(samples)
    pos_samples = assign_index(samples, pos_dictionary)
    neg_samples = assign_index(samples, neg_dictionary)
    del samples
    for i in range(0, len(pos_samples)):
        print(i)
        pos_sample = pos_samples[i]
        neg_sample = neg_samples[i]
        if pos_hmm.test(pos_sample) > neg_hmm.test(neg_sample):
            correct += 1
        total += 1

    # clean and test samples from negative data path
    samples = load_subdir(os.path.join(args.datapath, 'neg'))
    samples = clean_sentences(samples)
    pos_samples = assign_index(samples, pos_dictionary)
    neg_samples = assign_index(samples, neg_dictionary)
    del samples
    for i in range(0, len(neg_samples)):
        print(i)
        pos_sample = pos_samples[i]
        neg_sample = neg_samples[i]
        if pos_hmm.test(pos_sample) < neg_hmm.test(neg_sample):
            correct += 1
        total += 1

    # report accuracy  (no need for F1 on balanced data)
    print("%d/%d correct; accuracy %f" % (correct, total, correct / total))


if __name__ == '__main__':
    main()
