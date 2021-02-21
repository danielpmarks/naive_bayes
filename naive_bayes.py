# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018

"""
This is the main entry point for Part 1 of this MP. You should only modify code
within this file for Part 1 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np


def naiveBayes(train_set, train_labels, dev_set, smoothing_parameter, pos_prior):
    """
    train_set - List of list of words corresponding with each email
    example: suppose I had two emails 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two emails, first one was ham and second one was spam.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each email that we are testing on
              It follows the same format as train_set

    smoothing_parameter - The smoothing parameter --laplace (1.0 by default)
    pos_prior - positive prior probability (between 0 and 1)
    """

    # Set to keep track of the count of each word within the ham emails
    hamWords = {}
    hamWordCount = 0
    hamCount = 0
    # Set to keep track of the count of each word within the spam emails
    spamWords = {}
    spamWordCount = 0
    spamCount = 0

    # loop through each word tagged with a label
    i = 0
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
        else:
            spamCount += 1
            for word in train_set[i]:
                if word not in spamWords:
                    spamWords[word] = 1
                else:
                    spamWords[word] += 1

                spamWordCount += 1

    # The total count used to find P(type)
    totalEmails = hamCount + spamCount

    # Return list
    classification = []

    # Loop through each email
    for email in dev_set:
        # Use logarithms to prevent underflow

        # P(type = ham)
        p_type_ham = np.log(hamCount / totalEmails)
        # P(type = spam)
        p_type_spam = np.log(spamCount / totalEmails)

        ham_len = len(hamWords)
        spam_len = len(spamWords)

        for word in email:
            # Add the log probabilities given each type of email
            count_ham = 0
            if word in hamWords:
                count_ham = hamWords[word]
            p_type_ham += np.log((count_ham + smoothing_parameter) /
                                 (hamWordCount + smoothing_parameter*ham_len))
            count_spam = 0
            if word in spamWords:
                count_spam = spamWords[word]
            p_type_spam += np.log((count_spam + smoothing_parameter) /
                                  (spamWordCount + smoothing_parameter*spam_len))

        # factor in the prior probabilities
        p_type_ham += np.log(pos_prior)
        p_type_spam += np.log(1 - pos_prior)

        # compare probabilites and append to output
        if (p_type_ham > p_type_spam):
            classification.append(1)
        else:
            classification.append(0)

    return classification
