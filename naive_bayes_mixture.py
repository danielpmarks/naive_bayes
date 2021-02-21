# naive_bayes_mixture.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Modified by Jaewook Yeom 02/02/2020

"""
This is the main entry point for Part 2 of this MP. You should only modify code
within this file for Part 2 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
import numpy as np


def naiveBayesMixture(train_set, train_labels, dev_set, bigram_lambda, unigram_smoothing_parameter, bigram_smoothing_parameter, pos_prior):
    """
    train_set - List of list of words corresponding with each email
    example: suppose I had two emails 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two emails, first one was ham and second one was spam.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each email that we are testing on
              It follows the same format as train_set

    bigram_lambda - float between 0 and 1

    unigram_smoothing_parameter - Laplace smoothing parameter for unigram model (between 0 and 1)

    bigram_smoothing_parameter - Laplace smoothing parameter for bigram model (between 0 and 1)

    pos_prior - positive prior probability (between 0 and 1)
    """

    # Set to keep track of the count of each word within the ham emails
    hamWords = {}
    hamWordCount = 0

    # Ham bigram counters
    hamBigrams = {}
    hamBigramCount = 0

    # Set to keep track of the count of each word within the spam emails
    spamWords = {}
    spamWordCount = 0

    # Spam bigram counters
    spamBigrams = {}
    spamBigramCount = 0

    # Count number of each email type
    hamCount = 0
    spamCount = 0

    # loop through each word tagged with a label
    for i in range(len(train_set)):
        # increment the word count for each list
        if train_labels[i] == 1:
            hamCount += 1
            for word in train_set[i]:
                if word not in hamWords:
                    hamWords[word] = 1
                else:
                    hamWords[word] += 1

                hamWordCount += 1
            # Loop from 0 to len - 1 for bigrams
            for j in range(len(train_set[i])-1):
                bigram = train_set[i][j] + train_set[i][j+1]
                if bigram not in hamBigrams:
                    hamBigrams[bigram] = 1
                else:
                    hamBigrams[bigram] += 1

                hamBigramCount += 1

        else:
            spamCount += 1
            for word in train_set[i]:
                if word not in spamWords:
                    spamWords[word] = 1
                else:
                    spamWords[word] += 1

                spamWordCount += 1

            # Loop from 0 to len - 1 for bigrams
            for j in range(len(train_set[i])-1):
                bigram = train_set[i][j] + train_set[i][j+1]
                if bigram not in spamBigrams:
                    spamBigrams[bigram] = 1
                else:
                    spamBigrams[bigram] += 1

                spamBigramCount += 1

    # The total count used to find P(type)
    totalEmails = hamCount + spamCount

    # Return list
    classification = []

    # Loop through each email
    for email in dev_set:
        # Use logarithms to prevent underflow

        # P(type = ham)
        p_type_ham_words = np.log(hamCount / totalEmails)
        # P(type = spam)
        p_type_spam_words = np.log(spamCount / totalEmails)

        ham_words_len = len(hamWords)
        spam_words_len = len(spamWords)

        for word in email:
            # Add the log probabilities given each type of email
            count_ham = 0
            if word in hamWords:
                count_ham = hamWords[word]
            p_type_ham_words += np.log((count_ham + unigram_smoothing_parameter) /
                                       (hamWordCount + unigram_smoothing_parameter*ham_words_len))
            count_spam = 0
            if word in spamWords:
                count_spam = spamWords[word]
            p_type_spam_words += np.log((count_spam + unigram_smoothing_parameter) /
                                        (spamWordCount + unigram_smoothing_parameter*ham_words_len))

        # P(type = ham)
        p_type_ham_bigrams = np.log(hamCount / totalEmails)
        # P(type = spam)
        p_type_spam_bigrams = np.log(spamCount / totalEmails)

        ham_bigrams_len = len(hamBigrams)
        spam_bigrams_len = len(spamBigrams)

        for i in range(len(email) - 1):
            bigram = email[i] + email[i+1]
            count_ham = 0
            if bigram in hamBigrams:
                count_ham = hamBigrams[bigram]
            p_type_ham_bigrams += np.log((count_ham + bigram_smoothing_parameter) /
                                         (hamBigramCount + bigram_smoothing_parameter*ham_bigrams_len))
            count_spam = 0
            if bigram in spamBigrams:
                count_spam = spamBigrams[bigram]
            p_type_spam_bigrams += np.log((count_spam + bigram_smoothing_parameter) /
                                          (spamBigramCount + bigram_smoothing_parameter*spam_bigrams_len))

        # Combine word and bigram likelyhoods
        p_type_ham = (1 - bigram_lambda) * p_type_ham_words + \
            bigram_lambda * p_type_ham_bigrams
        p_type_spam = (1 - bigram_lambda) * p_type_spam_words + \
            bigram_lambda * p_type_spam_bigrams

        # factor in the prior probabilities
        p_type_ham += np.log(pos_prior)
        p_type_spam += np.log(1 - pos_prior)

        # compare probabilites and append to output
        if (p_type_ham > p_type_spam):
            classification.append(1)
        else:
            classification.append(0)

    return classification
